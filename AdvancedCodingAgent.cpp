#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <sys/wait.h>
#include <unistd.h>
#include <poll.h>
#include <signal.h>
#include <sys/resource.h>

using json = nlohmann::json;
namespace fs = std::filesystem;

struct TestCase {
    std::string id, name, description, task_prompt;
    std::vector<std::string> expected_outputs;
    std::vector<nlohmann::json> validation_checks;
    int max_turns;
    int timeout;
};

enum class ErrorCategory {
    NONE,
    TRANSIENT,
    QUOTA,
    PERMANENT
};

inline double get_time() {
    return std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

inline std::string md5_hash(const std::string &s) {
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : s) h = (h ^ c) * 1099511628211ULL;
    std::ostringstream oss;
    oss << std::hex << (h & 0xFFFFFFFFULL);
    return oss.str();
}

inline std::string error_type_from_output(const std::string &output) {
    std::string low = output;
    std::transform(low.begin(), low.end(), low.begin(), ::tolower);
    if (low.find("syntaxerror") != std::string::npos) return "SyntaxError";
    if (low.find("indentationerror") != std::string::npos) return "IndentationError";
    if (low.find("modulenotfounderror") != std::string::npos) return "ModuleNotFoundError";
    if (low.find("compilation failed") != std::string::npos) return "CompilationFailed";
    if (low.find("segmentation fault") != std::string::npos) return "SegmentationFault";
    return "GenericError";
}

static const std::unordered_set<std::string> STOP_WORDS = {
    "the", "and", "is", "in", "to", "of", "a", "for", "on", "with", "as", "by", "at", "an", "be",
    "this", "that", "it", "not", "or", "but", "are", "from", "has", "had", "have", "will", "would",
    "could", "should", "may", "can", "do", "does", "did", "was", "were", "been", "being", "am", "i",
    "you", "he", "she", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "its",
    "our", "their", "what", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "only", "own", "same", "so", "than", "through",
    "too", "under", "until", "up", "very", "with", "would"
};

inline std::vector<std::string> tokenize(const std::string& text) {
    std::string lower = text;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    static const std::regex re(R"(\b[a-z]{3,}\b)");
    std::sregex_iterator it(lower.begin(), lower.end(), re);
    std::sregex_iterator end;
    std::vector<std::string> tokens;
    for (; it != end; ++it) {
        std::string w = it->str();
        if (!STOP_WORDS.count(w)) {
            tokens.push_back(w);
        }
    }
    return tokens;
}

inline std::string extract_json_with_bracket_matching(const std::string & s) {
    int first_obj = -1;
    int last_obj = -1;
    int depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_string = false;
            }
        } else {
            if (c == '"') {
                in_string = true;
            } else if (c == '{') {
                if (depth == 0) {
                    first_obj = static_cast<int>(i);
                }
                depth++;
            } else if (c == '}') {
                if (depth > 0) {
                    depth--;
                    if (depth == 0) {
                        last_obj = static_cast<int>(i);
                    }
                }
            }
        }
    }
    if (first_obj != -1 && last_obj != -1 && last_obj > first_obj) {
        return s.substr(
            first_obj,
            last_obj - first_obj + 1
        );
    }
    return "";
}

inline std::string smart_sanitize_for_structured(const std::string &s) {
    std::string res;
    bool in_string = false;
    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (c == '"' && (i == 0 || s[i-1] != '\\')) {
            in_string = !in_string;
        }
        if (in_string && (c == '\n' || c == '\r')) {
            res += "\\n";
        } else {
            res += c;
        }
    }
    return res;
}

class RateLimiter {
public:
    explicit RateLimiter(int rpm = 60) : interval(60.0 / rpm) {}
    void acquire() {
        std::lock_guard<std::mutex> lk(mtx);
        double now = get_time();
        double sleep_time = last_call + interval - now;
        if (sleep_time > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(
                static_cast<long long>(sleep_time * 1000)));
        }
        last_call = get_time();
    }
private:
    double interval;
    double last_call = 0.0;
    std::mutex mtx;
};

inline size_t WriteCallback(void *contents, size_t size, size_t nmemb, std::string *out) {
    size_t total = size * nmemb;
    out->append(static_cast<char*>(contents), total);
    return total;
}

static int curl_abort_callback(void* clientp, curl_off_t /*dltotal*/, curl_off_t /*dlnow*/,
                               curl_off_t /*ultotal*/, curl_off_t /*ulnow*/) {
    const auto* flag = static_cast<const std::atomic<bool>*>(clientp);
    return (flag && flag->load(std::memory_order_relaxed)) ? 1 : 0;
}

inline bool http_post_json(const std::string &url,
                           const std::vector<std::string> &header_lines,
                           const json &payload,
                           int timeout_sec,
                           std::string &response,
                           std::string *error_out = nullptr,
                           long *http_code_out = nullptr,
                           const std::atomic<bool> *cancel_flag = nullptr) {
    CURL *curl = curl_easy_init();
    if (!curl) {
        if (error_out) *error_out = "curl_easy_init() failed";
        return false;
    }
    curl_slist *headers = nullptr;
    for (auto &h : header_lines)
        headers = curl_slist_append(headers, h.c_str());
    std::string body = payload.dump();
    response.clear();
    char curl_errbuf[CURL_ERROR_SIZE] = {0};
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 6L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout_sec);
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
    curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, curl_errbuf);
    if (cancel_flag) {
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
        curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, curl_abort_callback);
        curl_easy_setopt(curl, CURLOPT_XFERINFODATA,
                         const_cast<std::atomic<bool>*>(cancel_flag));
        curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    }
    CURLcode res = curl_easy_perform(curl);
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    if (http_code_out) *http_code_out = http_code;
    std::string error;
    if (res != CURLE_OK) {
        if (res == CURLE_ABORTED_BY_CALLBACK) {
            error = "Aborted by cancel flag";
        } else {
            error = std::string(curl_errbuf[0] ? curl_errbuf
                                               : curl_easy_strerror(res));
            error = "curl error: " + error;
        }
    } else if (http_code != 200) {
        error = "HTTP " + std::to_string(http_code) + ": " +
                response.substr(0, 300);
    }
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    if (!error.empty()) {
        if (error_out) *error_out = error;
        return false;
    }
    return true;
}

struct ChatResult {
    std::string content;
    bool ok = false;
    std::string error;
    ErrorCategory category = ErrorCategory::NONE;
};

class LLMClientBase {
public:
    explicit LLMClientBase(std::string api_key) : api_key(std::move(api_key)) {}
    virtual ~LLMClientBase() = default;
    void set_cancel_flag(std::atomic<bool>* flag) { cancel_flag_ = flag; }
    virtual ChatResult fetch(const std::string &model,
                             const std::vector<std::pair<std::string, std::string>> &messages,
                             double temperature,
                             int max_tokens,
                             const json &schema = json(),
                             int timeout_sec = 12) = 0;                             
    static ErrorCategory classify_error(long http_code, const std::string &err) {
        std::string low = err;
        std::transform(low.begin(), low.end(), low.begin(), ::tolower);
        if (http_code == 401 || http_code == 403 || http_code == 404 ||
            low.find("model_not_found") != std::string::npos ||
            low.find("does not exist") != std::string::npos ||
            low.find("invalid_api_key") != std::string::npos) {
            return ErrorCategory::PERMANENT;
        }
        if (http_code == 429 || http_code == 402 ||
            low.find("quota") != std::string::npos ||
            low.find("credits") != std::string::npos ||
            low.find("free-models-per-day") != std::string::npos ||
            low.find("rate limit") != std::string::npos) {
            return ErrorCategory::QUOTA;
        }
        if (http_code == 503 || http_code == 502 || http_code == 504 ||
            low.find("unavailable") != std::string::npos ||
            low.find("high demand") != std::string::npos) {
            return ErrorCategory::TRANSIENT;
        }
        return ErrorCategory::TRANSIENT;
    }
protected:
    std::string api_key;
    std::atomic<bool>* cancel_flag_ = nullptr;
    static json infer_google_schema_type(const json &value) {
        if (value.is_string()) return {{"type", "STRING"}};
        if (value.is_number()) return {{"type", "NUMBER"}};
        if (value.is_boolean()) return {{"type", "BOOLEAN"}};
        if (value.is_array()) {
            json item_schema = value.empty()
                ? json::object({{"type", "STRING"}})
                : infer_google_schema_type(value[0]);
            return {{"type", "ARRAY"}, {"items", item_schema}};
        }
        if (value.is_object()) {
            json obj;
            obj["type"] = "OBJECT";
            obj["properties"] = json::object();
            for (auto &[k, v] : value.items()) {
                obj["properties"][k] = infer_google_schema_type(v);
            }
            return obj;
        }
        return {{"type", "STRING"}};
    }
    static json to_google_response_schema(const json &schema) {
        if (schema.is_null() || schema.empty()) return json::object();
        if (schema.is_object() && schema.contains("type") && schema.contains("properties")) return schema;
        json result = json::object();
        result["type"] = "OBJECT";
        result["properties"] = json::object();
        if (schema.is_object()) {
            for (auto &[key, value] : schema.items()) {
                result["properties"][key] = infer_google_schema_type(value);
            }
        }
        return result;
    }
    static std::string extract_json_payload(std::string s) {
        auto trim = [](std::string &str) {
            while (!str.empty() && std::isspace(static_cast<unsigned char>(str.front()))) str.erase(str.begin());
            while (!str.empty() && std::isspace(static_cast<unsigned char>(str.back()))) str.pop_back();
        };
        size_t first_fence = s.find("```");
        if (first_fence != std::string::npos) {
            size_t first_nl = s.find('\n', first_fence);
            if (first_nl != std::string::npos) {
                size_t last_fence = s.rfind("```");
                if (last_fence != std::string::npos && last_fence > first_nl) {
                    std::string extracted = s.substr(first_nl + 1, last_fence - first_nl - 1);
                    trim(extracted);
                    if (!extracted.empty() && (extracted.front() == '{' || extracted.front() == '[')) {
                        return extracted;
                    }
                }
            }
        }
        size_t start = std::string::npos;
        for (size_t i = 0; i < s.size(); ++i) {
            if (s[i] == '{' || s[i] == '[') {
                start = i;
                break;
            }
        }
        if (start != std::string::npos) {
            char open_char = s[start];
            char close_char = (open_char == '{') ? '}' : ']';
            int depth = 0;
            bool in_string = false;
            bool escape = false;
            size_t end = std::string::npos;
            for (size_t i = start; i < s.size(); ++i) {
                if (escape) {
                    escape = false;
                    continue;
                }
                if (s[i] == '\\') {
                    escape = true;
                    continue;
                }
                if (s[i] == '"') {
                    in_string = !in_string;
                    continue;
                }
                if (!in_string) {
                    if (s[i] == open_char) {
                        depth++;
                    } else if (s[i] == close_char) {
                        depth--;
                        if (depth == 0) {
                            end = i;
                            break;
                        }
                    }
                }
            }
            if (end != std::string::npos) {
                return s.substr(start, end - start + 1);
            }
        }
        trim(s);
        return s;
    }
};

class GroqClient : public LLMClientBase {
public:
    explicit GroqClient(const std::string &key) : LLMClientBase(key) {}
    ChatResult fetch(const std::string &model,
                     const std::vector<std::pair<std::string, std::string>> &messages,
                     double temperature,
                     int max_tokens,
                     const json &schema = json(), int timeout_sec = 12) override {
        const bool wants_structured = !schema.is_null() && !schema.empty();
        const bool model_supports_json_mode = model_supports_forced_json(model);
        const bool schema_is_simple = is_simple_schema(schema);
        const bool use_forced_json =
            wants_structured && model_supports_json_mode && schema_is_simple;
        json msgs = json::array();
        bool has_system = false;
        for (const auto &[role, content] : messages) {
            if (role == "system") has_system = true;
            msgs.push_back({{"role", role}, {"content", content}});
        }
        if (wants_structured && !use_forced_json) {
            const std::string json_hint =
                "You must reply with a single valid JSON object and nothing else. "
                "No markdown, no code fences, no commentary. "
                "The JSON must be parseable by a strict parser.";
            if (has_system) {
                for (auto &m : msgs) {
                    if (m["role"] == "system") {
                        m["content"] = json_hint + "\n\n" + m["content"].get<std::string>();
                        break;
                    }
                }
            } else {
                msgs.insert(msgs.begin(), {{"role", "system"}, {"content", json_hint}});
            }
        }
        json payload = {
            {"model", model},
            {"temperature", temperature},
            {"max_tokens", max_tokens},
            {"messages", msgs}
        };
        if (use_forced_json) {
            payload["response_format"] = {{"type", "json_object"}};
        }
        std::string resp, err_out;
        long http_code = 0;
        if (!http_post_json("https://api.groq.com/openai/v1/chat/completions",
                            {"Authorization: Bearer " + api_key,
                             "Content-Type: application/json"},
                            payload, timeout_sec, resp, &err_out, &http_code,
                            cancel_flag_)) {
            ErrorCategory cat = classify_error(http_code, err_out);
            if (http_code == 400 &&
                (err_out.find("json_validate_failed") != std::string::npos ||
                 err_out.find("Failed to validate JSON") != std::string::npos)) {
                cat = ErrorCategory::TRANSIENT;
            }
            return {"", false, err_out, cat};
        }
        try {
            json data = json::parse(resp);
            if (data.contains("error") && data["error"].is_object() &&
                data["error"].contains("message")) {
                std::string msg = data["error"]["message"].get<std::string>();
                ErrorCategory cat = classify_error(http_code, msg);
                if (msg.find("json_validate_failed") != std::string::npos ||
                    msg.find("Failed to validate JSON") != std::string::npos) {
                    cat = ErrorCategory::TRANSIENT;
                }
                return {"", false, msg, cat};
            }
            if (data.contains("choices") && data["choices"].is_array() &&
                !data["choices"].empty()) {
                auto &choice = data["choices"][0];
                if (choice.contains("message") &&
                    choice["message"].contains("content") &&
                    choice["message"]["content"].is_string()) {
                    std::string content =
                        choice["message"]["content"].get<std::string>();
                    if (wants_structured) {
                        content = extract_json_payload(content);
                    }
                    return {content, true, "", ErrorCategory::NONE};
                }
            }
            return {"", false, "Unexpected response structure",
                    ErrorCategory::TRANSIENT};
        } catch (const std::exception &e) {
            return {"", false, std::string("JSON parse error: ") + e.what(),
                    ErrorCategory::TRANSIENT};
        }
    }
private:
    static bool model_supports_forced_json(const std::string &model) {
        static const std::vector<std::string> reliable = {
            "openai/gpt-oss", "qwen/qwen3", "llama-3.1-8b-instant",
            "llama-3.3-70b", "llama-3.1-70b", "gemma2-9b-it",
            "mixtral-8x7b"
        };
        for (const auto &m : reliable) {
            if (model.find(m) != std::string::npos) return true;
        }
        return false;
    }
    static bool is_simple_schema(const json &schema) {
        if (schema.is_null() || schema.empty()) return true;
        std::string dumped = schema.dump();
        if (dumped.find("\"oneOf\"") != std::string::npos) return false;
        if (dumped.find("\"anyOf\"") != std::string::npos) return false;
        if (dumped.find("\"allOf\"") != std::string::npos) return false;
        if (dumped.size() > 2500) return false;
        return true;
    }
};

class OpenRouterClient : public LLMClientBase {
public:
    OpenRouterClient(const std::string &key, int rpm = 30)
        : LLMClientBase(key), limiter(rpm) {}
    ChatResult fetch(const std::string &model,
                     const std::vector<std::pair<std::string, std::string>> &messages,
                     double temperature,
                     int max_tokens,
                     const json &schema = json(), int timeout_sec = 12) override {
        limiter.acquire();
        const bool wants_structured = !schema.is_null() && !schema.empty();
        const bool use_forced_json =
            wants_structured && model_supports_forced_json(model);
        json msgs = json::array();
        bool has_system = false;
        for (const auto &[role, content] : messages) {
            if (role == "system") has_system = true;
            msgs.push_back({{"role", role}, {"content", content}});
        }
        if (wants_structured && !use_forced_json) {
            const std::string json_hint =
                "Reply with a single valid JSON object only. "
                "No markdown fences, no commentary, no extra text. "
                "The response must be directly parseable by json.loads / nlohmann::json.";
            if (has_system) {
                for (auto &m : msgs) {
                    if (m["role"] == "system") {
                        m["content"] = json_hint + "\n\n" + m["content"].get<std::string>();
                        break;
                    }
                }
            } else {
                msgs.insert(msgs.begin(), {{"role", "system"}, {"content", json_hint}});
            }
        }
        json payload = {
            {"model", model},
            {"temperature", temperature},
            {"max_tokens", max_tokens},
            {"messages", msgs}
        };
        if (use_forced_json) {
            payload["response_format"] = {{"type", "json_object"}};
        }
        payload["provider"] = {
            {"allow_fallbacks", true},
            {"require_parameters", true}
        };
        std::string resp, err_out;
        long http_code = 0;
        if (!http_post_json("https://openrouter.ai/api/v1/chat/completions",
                            {"Authorization: Bearer " + api_key,
                             "Content-Type: application/json",
                             "HTTP-Referer: https://localhost",
                             "X-Title: CodingAgent"},
                            payload, timeout_sec, resp, &err_out, &http_code,
                            cancel_flag_)) {
            return make_error_result(http_code, err_out, resp);
        }
        try {
            json data = json::parse(resp);
            if (data.contains("error") && data["error"].is_object()) {
                std::string msg;
                if (data["error"].contains("message") && data["error"]["message"].is_string()) {
                    msg = data["error"]["message"].get<std::string>();
                } else {
                    msg = data["error"].dump();
                }
                return make_error_result(http_code, msg, resp);
            }
            if (data.contains("choices") && data["choices"].is_array() && !data["choices"].empty()) {
                auto &choice = data["choices"][0];
                if (choice.contains("message") &&
                    choice["message"].contains("content") &&
                    choice["message"]["content"].is_string()) {
                    std::string content = choice["message"]["content"].get<std::string>();
                    if (wants_structured) {
                        content = extract_json_payload(content);
                    }
                    return {content, true, "", ErrorCategory::NONE};
                }
            }
            if (wants_structured) {
                std::string fallback = extract_json_fallback(data);
                if (!fallback.empty()) {
                    return {fallback, true, "", ErrorCategory::NONE};
                }
            }
            return {"", false, "Unexpected response structure", ErrorCategory::TRANSIENT};
        } catch (const std::exception &e) {
            return {"", false, std::string("JSON parse error: ") + e.what(), ErrorCategory::TRANSIENT};
        }
    }
private:
    RateLimiter limiter;
    static bool model_supports_forced_json(const std::string &model) {
        static const std::vector<std::string> reliable = {
            "openai/", "anthropic/", "google/gemini", "google/gemma",
            "meta-llama/llama-3", "meta-llama/llama-3.1", "meta-llama/llama-3.3",
            "mistralai/", "qwen/", "deepseek/", "nvidia/nemotron",
            "thinkingmachines/", "cohere/", "xingchen-agi/"
        };
        for (const auto &prefix : reliable) {
            if (model.find(prefix) != std::string::npos) return true;
        }
        return false;
    }
    ChatResult make_error_result(long http_code, const std::string &err_out, const std::string &raw_body) const {
        std::string msg = err_out;
        ErrorCategory cat = classify_error(http_code, err_out);
        try {
            json body = json::parse(raw_body.empty() ? err_out : raw_body);
            if (body.contains("error") && body["error"].is_object()) {
                auto &e = body["error"];
                if (e.contains("message") && e["message"].is_string()) {
                    msg = e["message"].get<std::string>();
                }
                if (msg.find("free-models-per-day") != std::string::npos ||
                    msg.find("Rate limit exceeded") != std::string::npos) {
                    cat = ErrorCategory::QUOTA;
                }
            }
        } catch (...) {}
        if (msg.find("json_validate") != std::string::npos ||
            msg.find("Failed to validate JSON") != std::string::npos) {
            cat = ErrorCategory::TRANSIENT;
        }
        return {"", false, msg, cat};
    }
    static std::string extract_json_fallback(const json &data) {
        if (data.is_object() && data.contains("content") && data["content"].is_string()) {
            return extract_json_payload(data["content"].get<std::string>());
        }
        if (data.is_object() && data.contains("tool")) {
            return data.dump();
        }
        return "";
    }
};

class GoogleClient : public LLMClientBase {
public:
    explicit GoogleClient(const std::string &key) : LLMClientBase(key) {}
    ChatResult fetch(const std::string &model,
                     const std::vector<std::pair<std::string, std::string>> &messages,
                     double temperature,
                     int max_tokens,
                     const json &schema = json(), int timeout_sec = 12) override {
        std::string url = "https://generativelanguage.googleapis.com/v1beta/models/" + model + ":generateContent?key=" + api_key;
        const bool wants_structured = !schema.is_null() && !schema.empty();
        const bool use_full_schema = wants_structured && is_schema_safe_for_google(schema);
        json contents = json::array();
        json sys_parts = json::array();
        for (const auto &[role, text] : messages) {
            if (role == "system") {
                sys_parts.push_back({{"text", text}});
            } else {
                std::string mapped = (role == "user") ? "user" : "model";
                if (!contents.empty() && contents.back()["role"] == mapped) {
                    contents.back()["parts"][0]["text"] =
                        contents.back()["parts"][0]["text"].get<std::string>() + "\n\n" + text;
                } else {
                    json item = json::object();
                    item["role"] = mapped;
                    item["parts"] = json::array({json{{"text", text}}});
                    contents.push_back(item);
                }
            }
        }
        if (contents.empty()) {
            contents.push_back({{"role", "user"}, {"parts", json::array({json{{"text", "Begin task."}}})}});
        }
        if (wants_structured && !use_full_schema) {
            sys_parts.push_back({
                {"text", "You must reply with a single valid JSON object only. "
                         "No markdown fences, no commentary, no extra text. "
                         "The response must be directly parseable by a strict JSON parser."}
            });
        }
        json genConfig = {
            {"temperature", temperature},
            {"maxOutputTokens", max_tokens}
        };
        if (wants_structured) {
            genConfig["responseMimeType"] = "application/json";
            if (use_full_schema) {
                genConfig["responseSchema"] = to_google_response_schema(schema);
            }
        }
        json payload = {
            {"contents", contents},
            {"generationConfig", genConfig}
        };
        if (!sys_parts.empty()) {
            payload["systemInstruction"] = {{"parts", sys_parts}};
        }
        std::string resp, err_out;
        long http_code = 0;
        if (!http_post_json(url, {"Content-Type: application/json"}, payload, timeout_sec, resp, &err_out, &http_code, cancel_flag_)) {
            return make_error_result(http_code, err_out, resp);
        }
        try {
            json data = json::parse(resp);
            if (data.contains("error") && data["error"].is_object() && data["error"].contains("message")) {
                std::string msg = data["error"]["message"].get<std::string>();
                return make_error_result(http_code, msg, resp);
            }
            if (data.contains("promptFeedback") && data["promptFeedback"].contains("blockReason")) {
                std::string reason = data["promptFeedback"]["blockReason"].get<std::string>();
                return {"", false, "Prompt blocked by safety filter: " + reason, ErrorCategory::PERMANENT};
            }
            if (!data.contains("candidates") || !data["candidates"].is_array() || data["candidates"].empty()) {
                return {"", false, "No candidates returned", ErrorCategory::TRANSIENT};
            }
            auto &cand = data["candidates"][0];
            if (cand.contains("finishReason") && cand["finishReason"].is_string()) {
                std::string fr = cand["finishReason"].get<std::string>();
                if (fr == "SAFETY" || fr == "RECITATION" || fr == "OTHER") {
                    return {"", false, "Generation stopped: " + fr,
                            fr == "SAFETY" ? ErrorCategory::PERMANENT : ErrorCategory::TRANSIENT};
                }
            }
            if (cand.contains("content") && cand["content"].contains("parts") &&
                cand["content"]["parts"].is_array() && !cand["content"]["parts"].empty()) {
                std::string full_text;
                for (auto &part : cand["content"]["parts"]) {
                    if (part.contains("text") && part["text"].is_string()) {
                        full_text += part["text"].get<std::string>();
                    }
                }
                if (!full_text.empty()) {
                    if (wants_structured) {
                        full_text = extract_json_payload(full_text);
                    }
                    return {full_text, true, "", ErrorCategory::NONE};
                }
            }
            return {"", false, "Empty candidate content", ErrorCategory::TRANSIENT};
        } catch (const std::exception &e) {
            return {"", false, std::string("JSON parse error: ") + e.what(), ErrorCategory::TRANSIENT};
        }
    }
private:
    ChatResult make_error_result(long http_code, const std::string &err_out, const std::string &raw_body) const {
        return {"", false, err_out, classify_error(http_code, err_out)};
    }
    static bool is_schema_safe_for_google(const json &schema) {
        if (schema.is_null() || schema.empty()) return false;
        std::string dumped = schema.dump();
        if (dumped.find("\"anyOf\"") != std::string::npos ||
            dumped.find("\"oneOf\"") != std::string::npos ||
            dumped.find("\"allOf\"") != std::string::npos) return false;
        return true;
    }
};

class LLMBackendManager {
public:
    struct Config {
        std::map<std::string, std::string> models;
        std::vector<std::string> priority;
        int global_rpm = 30;
    };
    std::unordered_set<std::string> dead_models_;
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> cooldown_;
    std::mutex health_mu_;
    LLMBackendManager(const Config& cfg) : config(cfg), limiter(cfg.global_rpm) {
        uint64_t seed = std::chrono::steady_clock::now().time_since_epoch().count();
        rng_.seed(seed);
        const char* groq = getenv("GROQ_API_KEY");
        const char* openrouter = getenv("OPENROUTER_API_KEY");
        const char* google = getenv("GOOGLE_API_KEY");
        if (groq) {
            auto c = std::make_shared<GroqClient>(groq);
            c->set_cancel_flag(&aborted);
            clients["groq"] = c;
        }
        if (openrouter) {
            auto c = std::make_shared<OpenRouterClient>(openrouter);
            c->set_cancel_flag(&aborted);
            clients["openrouter"] = c;
        }
        if (google) {
            auto c = std::make_shared<GoogleClient>(google);
            c->set_cancel_flag(&aborted);
            clients["google"] = c;
        }        
        fallback_models_["groq"] = {
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
            "llama-3.1-8b-instant"
        };
        fallback_models_["openrouter"] = {
            "openrouter/free",
            "thinkingmachines/inkling-small:free",
            "google/gemma-4-31b-it:free",
            "nvidia/nemotron-3.5-lightning:free",
            "nvidia/nemotron-3-super-120b-a12b:free",
            "nvidia/nemotron-3-ultra-550b-a55b:free",
            "xingchen-agi/xing4.0-29b-a4b:free",
            "cohere/north-mini-code:free",
            "google/gemma-4-26b-a4b-it:free"
        };
        fallback_models_["google"] = {
            "gemini-3.8-flash",
            "gemini-3.6-flash",
            "gemini-3.5-flash-lite",
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash"
        };
        for (auto& b : config.priority) {
            if (clients.count(b)) {
                current_backend = b;
                break;
            }
        }
        if (current_backend.empty()) {
            for (auto& [b, _] : clients) {
                current_backend = b;
                break;
            }
        }
        if (current_backend.empty())
            throw std::runtime_error("No LLM backend available");
        for (const auto& [b_name, _] : clients) {
            std::vector<std::string> m_list;
            if (config.models.count(b_name)) {
                m_list.push_back(config.models.at(b_name));
            }
            if (fallback_models_.count(b_name)) {
                for (const auto& m : fallback_models_[b_name]) {
                    if (std::find(m_list.begin(), m_list.end(), m) == m_list.end()) {
                        m_list.push_back(m);
                    }
                }
            }
            active_models_[b_name] = m_list;
            auto& st = backend_stats[b_name];
            st.alpha = 2.0;
            st.beta = 1.0;
            st.ewma_latency = 1.0;
            for (size_t i = 0; i < config.priority.size(); ++i) {
                if (config.priority[i] == b_name) {
                    st.alpha += static_cast<double>(config.priority.size() - i) * 2.0;
                    break;
                }
            }
        }
    }
    void abort() { aborted = true; }
    void reset_abort() { aborted = false; }
    long long get_logical_call_count() const {
        return logical_call_count_.load(std::memory_order_relaxed);
    }
    void reset_cooldowns() {
        std::lock_guard<std::mutex> lk(state_mtx);
        cooldown_until.clear();
    }
    static double extract_retry_delay_seconds(const std::string& err) {
        std::string low = err;
        std::transform(low.begin(), low.end(), low.begin(), ::tolower);
        static const std::regex re_retry(
            R"((?:retry|try again|please retry|wait)\s+(?:in\s+)?(\d+(?:\.\d+)?)\s?(ms|s|sec|secs|second|seconds|m|min|mins|minute|minutes|h|hr|hrs|hour|hours)?)",
            std::regex::icase);
        static const std::regex re_reset(R"(reset\s=\s(\d+))");
        double best = 0.0;
        std::smatch m;
        if (std::regex_search(low, m, re_retry) && m.size() >= 2) {
            double val = std::stod(m[1].str());
            std::string unit = m.size() > 2 ? m[2].str() : "";
            std::transform(unit.begin(), unit.end(), unit.begin(), ::tolower);
            if (unit == "ms") {
                best = std::max(best, val / 1000.0);
            } else if (unit.empty() || unit[0] == 's') {
                best = std::max(best, val);
            } else if (unit[0] == 'm') {
                best = std::max(best, val * 60.0);
            } else if (unit[0] == 'h') {
                best = std::max(best, val * 3600.0);
            }
        }
        if (std::regex_search(err, m, re_reset) && m.size() >= 2) {
            try {
                double epoch_ms = std::stod(m[1].str());
                double now_ms = get_time() * 1000.0;
                if (epoch_ms > now_ms)
                    best = std::max(best, (epoch_ms - now_ms) / 1000.0);
            } catch (...) {}
        }
        return best;
    }
    void set_cooldown(const std::string& backend, ErrorCategory category,
                      const std::string& error_msg = "") {
        std::lock_guard<std::mutex> lk(state_mtx);
        const double now = get_time();
        const double duration = (category == ErrorCategory::PERMANENT) ? 120.0 : 3.0;
        cooldown_until[backend] = now + duration;
        const char* tag = (category == ErrorCategory::PERMANENT) ? "PERMANENT"
                         : (category == ErrorCategory::QUOTA)     ? "QUOTA"
                                                                  : "TRANSIENT";
        std::ostringstream oss;
        oss << "[LLM] cooldown backend=" << backend
            << " category=" << tag
            << " until=+" << duration << "s";
        if (!error_msg.empty()) {
            const std::string& shown = error_msg.size() > 160
                                     ? error_msg.substr(0, 160) + "…"
                                     : error_msg;
            oss << " reason=\"" << shown << "\"";
        }
        std::cerr << oss.str() << "\n";
    }
    ChatResult chat(const std::vector<std::pair<std::string, std::string>>& messages,
                    double temperature = 0.0,
                    int max_tokens = 2048,
                    const json& schema = json()) {
        logical_call_count_.fetch_add(1, std::memory_order_relaxed);
        if (aborted) return {"", false, "Aborted by agent.", ErrorCategory::PERMANENT};
        const double deadline = get_time() + kChatDeadlineSec;
        limiter.acquire();
        if (aborted) return {"", false, "Aborted by agent.", ErrorCategory::PERMANENT};
        const int max_retry_passes = 2;
        for (int pass = 0; pass <= max_retry_passes; ++pass) {
            if (aborted) return {"", false, "Aborted by agent.", ErrorCategory::PERMANENT};
            if (get_time() > deadline) return {"", false, "Deadline exceeded", ErrorCategory::TRANSIENT};
            std::vector<std::string> order = get_backend_order();
            for (const auto& backend : order) {
                if (aborted) return {"", false, "Aborted by agent.", ErrorCategory::PERMANENT};
                if (get_time() > deadline) return {"", false, "Deadline exceeded", ErrorCategory::TRANSIENT};
                if (is_on_cooldown(backend)) continue;
                auto it = clients.find(backend);
                if (it == clients.end()) continue;
                std::vector<std::string> model_list = get_active_models_sorted(backend);
                bool any_usable = false;
                for (const auto& m : model_list) {
                    if (model_usable_now(backend, m)) {
                        any_usable = true;
                        break;
                    }
                }
                if (!any_usable) continue;
                bool backend_failed = false;
                int transient_failures = 0;
                ErrorCategory worst_category = ErrorCategory::NONE;
                std::string worst_error;
                int models_remaining = static_cast<int>(model_list.size());
                for (const auto& model : model_list) {
                    if (aborted) return {"", false, "Aborted by agent.", ErrorCategory::PERMANENT};
                    if (get_time() > deadline) return {"", false, "Deadline exceeded", ErrorCategory::TRANSIENT};
                    if (!model_usable_now(backend, model)) {
                        models_remaining--;
                        continue;
                    }
                    double remaining = deadline - get_time();
                    if (remaining < 1.5) break;
                    double ewma = get_model_ewma_latency(backend, model);
                    double fair_share = remaining / std::max(1, models_remaining);
                    double budget = std::max(2.0, std::min({ewma * 2.5, fair_share, kMaxAttemptSec}));
                    if (budget < 2.0) {
                        models_remaining--;
                        continue;
                    }
                    int timeout_int = static_cast<int>(std::max(1.0, budget));
                    double t_start = get_time();
                    ChatResult r = it->second->fetch(model, messages, temperature, max_tokens, schema, timeout_int);
                    double elapsed = get_time() - t_start;
                    models_remaining--;
                    if (r.ok) {
                        record_model_success(backend, model, elapsed);
                        clear_backend_cooldown(backend);
                        record_backend_success(backend, elapsed);
                        {
                            std::lock_guard<std::mutex> lk(state_mtx);
                            current_backend = backend;
                        }
                        return r;
                    }
                    if (aborted) {
                        return {"", false, "Aborted by agent.", ErrorCategory::PERMANENT};
                    }
                    record_model_failure(backend, model, r.category, r.error);
                    std::cerr << "[LLM] backend=" << backend << " model=" << model
                              << " error=" << r.error << " category=" << static_cast<int>(r.category)
                              << " elapsed=" << elapsed << "s\n";
                    if (is_model_not_found_error(r.category, r.error)) {
                        prune_model(backend, model);
                        continue;
                    }
                    if (r.category == ErrorCategory::PERMANENT || r.category == ErrorCategory::QUOTA) {
                        record_backend_failure(backend);
                        if (worst_category == ErrorCategory::NONE ||
                            (r.category == ErrorCategory::PERMANENT && worst_category != ErrorCategory::PERMANENT)) {
                            worst_category = r.category;
                            worst_error = r.error;
                        } else if (r.category == worst_category) {
                            worst_error = r.error;
                        }
                        continue;
                    }
                    transient_failures++;
                    if (worst_category == ErrorCategory::NONE) {
                        worst_category = ErrorCategory::TRANSIENT;
                        worst_error = r.error;
                    }
                    if (transient_failures >= 2) {
                        record_backend_failure(backend);
                        set_cooldown(backend, ErrorCategory::TRANSIENT, "Too many transient failures");
                        backend_failed = true;
                        break;
                    }
                }
                if (!backend_failed) {
                    if (worst_category == ErrorCategory::PERMANENT) {
                        set_cooldown(backend, ErrorCategory::PERMANENT, worst_error);
                    } else if (transient_failures > 0) {
                        set_cooldown(backend, ErrorCategory::TRANSIENT, worst_error);
                    }
                }
            }
            if (pass == max_retry_passes) break;
            double wait_time = min_cooldown_remaining();
            double remaining = deadline - get_time();
            if (wait_time <= 0.0 || wait_time == std::numeric_limits<double>::infinity() ||
                wait_time > 30.0 || wait_time > remaining) {
                break;
            }
            double sleep_increment = 0.1;
            double slept = 0.0;
            while (slept < wait_time && !aborted) {
                double step = std::min(sleep_increment, wait_time - slept);
                std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<long long>(step * 1000)));
                slept += step;
            }
        }
        return {"", false, "All backends exhausted.", ErrorCategory::PERMANENT};
    }
    json structured_chat(const std::vector<std::pair<std::string, std::string>>& messages,
                         const json& schema,
                         double temperature = 0.0,
                         int max_tokens = 512) {
        std::string system =
            "You are a precise JSON-only assistant. Output ONLY valid JSON matching the schema below.\n"
            "Schema:\n" + schema.dump(2) + "\nOutput the raw JSON immediately without markdown backticks.";
        std::vector<std::pair<std::string, std::string>> full = {{"system", system}};
        full.insert(full.end(), messages.begin(), messages.end());
        ChatResult r = chat(full, temperature, max_tokens, schema);
        if (!r.ok) return json::object();
        std::string content = r.content;
        size_t pos = 0;
        while ((pos = content.find("```")) != std::string::npos) {
            size_t end = content.find("\n", pos + 3);
            if (end == std::string::npos) end = content.size();
            content.erase(pos, end - pos + 1);
        }
        std::string extracted = extract_json_with_bracket_matching(content);
        content = extracted.empty() ? content : extracted;
        try {
            return json::parse(content);
        } catch (...) {
            std::string recovered = extract_json_with_bracket_matching(
                smart_sanitize_for_structured(r.content));
            if (!recovered.empty()) {
                try {
                    return json::parse(recovered);
                } catch (...) {}
            }
        }
        return json::object();
    }
    void clear_backend_cooldown(const std::string& backend) {
        std::lock_guard<std::mutex> lk(state_mtx);
        cooldown_until.erase(backend);
    }
    double min_cooldown_remaining() {
        std::lock_guard<std::mutex> lk(state_mtx);
        double now = get_time();
        double min_wait = std::numeric_limits<double>::infinity();
        for (auto& [_, until] : cooldown_until) {
            if (until > now && until != std::numeric_limits<double>::infinity()) {
                min_wait = std::min(min_wait, until - now);
            }
        }
        return min_wait == std::numeric_limits<double>::infinity() ? 0.0 : min_wait;
    }
    bool model_is_usable(const std::string& key) {
        std::lock_guard<std::mutex> lk(health_mu_);
        if (dead_models_.count(key)) return false;
        auto it = cooldown_.find(key);
        if (it == cooldown_.end()) return true;
        if (std::chrono::steady_clock::now() < it->second) return false;
        cooldown_.erase(it);
        return true;
    }
    void report_model_failure(const std::string& key,
                              int category,
                              const std::string& message) {
        std::lock_guard<std::mutex> lk(health_mu_);
        if (category == 3) {
            dead_models_.insert(key);
            return;
        }
        double wait_s = 20.0;
        std::smatch m;
        if (std::regex_search(message, m,
                              std::regex(R"(try again in (?:(\d+)m)?([\d.]+)s)"))) {
            double mins = m[1].matched ? std::stod(m[1].str()) : 0.0;
            wait_s = mins * 60.0 + std::stod(m[2].str());
        }
        cooldown_[key] = std::chrono::steady_clock::now() +
                         std::chrono::milliseconds(static_cast<long>(wait_s * 1000));
    }

private:
    static constexpr double kChatDeadlineSec = 30.0;
    static constexpr double kMaxAttemptSec = 20.0;
    struct BackendStats {
        double alpha = 1.0;
        double beta = 1.0;
        double ewma_latency = 1.0;
        bool initialized = false;
        double last_success_time = 0.0;
        int total_calls = 0;
    };
    struct ModelHealth {
        int consecutive_failures = 0;
        double cooldown_until = 0.0;
        bool blacklisted = false;
        double ewma_latency = 2.0;
    };
    Config config;
    std::map<std::string, std::shared_ptr<LLMClientBase>> clients;
    std::string current_backend;
    std::map<std::string, double> cooldown_until;
    std::unordered_map<std::string, BackendStats> backend_stats;
    std::unordered_map<std::string, std::vector<std::string>> active_models_;
    mutable std::mutex state_mtx;
    RateLimiter limiter;
    std::atomic<bool> aborted{false};
    std::atomic<long long> logical_call_count_{0};
    std::mt19937 rng_;
    std::unordered_map<std::string, std::vector<std::string>> fallback_models_;
    std::unordered_map<std::string, ModelHealth> model_health_;
    std::mutex health_mtx_;
    static std::string model_key(const std::string& backend, const std::string& model) {
        return backend + "\n" + model;
    }
    static bool is_permanently_dead_model(ErrorCategory category, const std::string& err) {
        if (category != ErrorCategory::PERMANENT) return false;
        std::string low = err;
        std::transform(low.begin(), low.end(), low.begin(), ::tolower);
        return low.find("model_not_found") != std::string::npos ||
               low.find("does not exist") != std::string::npos ||
               low.find("not found") != std::string::npos ||
               low.find("unavailable") != std::string::npos ||
               low.find("invalid model") != std::string::npos ||
               low.find("decommission") != std::string::npos ||
               low.find("unsupported model") != std::string::npos;
    }
    bool model_usable_now(const std::string& backend, const std::string& model) {
        std::lock_guard<std::mutex> lk(health_mtx_);
        auto it = model_health_.find(model_key(backend, model));
        if (it == model_health_.end()) return true;
        const auto& h = it->second;
        if (h.blacklisted) return false;
        return get_time() >= h.cooldown_until;
    }
    double get_model_ewma_latency(const std::string& backend, const std::string& model) {
        std::lock_guard<std::mutex> lk(health_mtx_);
        auto it = model_health_.find(model_key(backend, model));
        if (it == model_health_.end()) return 2.0;
        return it->second.ewma_latency;
    }
    void record_model_success(const std::string& backend, const std::string& model,
                              double elapsed_sec) {
        std::lock_guard<std::mutex> lk(health_mtx_);
        auto& h = model_health_[model_key(backend, model)];
        h.consecutive_failures = 0;
        h.cooldown_until = 0.0;
        h.ewma_latency = 0.7 * h.ewma_latency + 0.3 * std::max(0.1, elapsed_sec);
    }
    void record_model_failure(const std::string& backend, const std::string& model,
                              ErrorCategory category, const std::string& err) {
        bool should_prune = false;
        {
            std::lock_guard<std::mutex> lk(health_mtx_);
            const std::string key = model_key(backend, model);
            auto& h = model_health_[key];
            h.consecutive_failures++;
            h.ewma_latency = std::min(h.ewma_latency * 1.3, 30.0);
            if (is_permanently_dead_model(category, err)) {
                h.blacklisted = true;
                should_prune = true;
            } else if (category == ErrorCategory::PERMANENT) {
                h.cooldown_until = get_time() + 120.0;
            } else {
                double delay =
                    8.0 * static_cast<double>(1 << std::min(h.consecutive_failures - 1, 4));
                if (category == ErrorCategory::QUOTA) {
                    double suggested = extract_retry_delay_seconds(err);
                    if (suggested > 0.0) delay = std::max(delay, std::min(suggested, 120.0));
                }
                h.cooldown_until = get_time() + std::min(delay, 120.0);
            }
        }
        if (should_prune) {
            prune_model(backend, model);
        }
    }
    void record_backend_success(const std::string& backend, double elapsed_sec = 0.0) {
        std::lock_guard<std::mutex> lk(state_mtx);
        auto& st = backend_stats[backend];
        st.alpha = st.alpha * 0.95 + 1.0;
        st.total_calls++;
        if (elapsed_sec > 0.0) {
            st.ewma_latency = 0.7 * st.ewma_latency + 0.3 * elapsed_sec;
        }
        st.last_success_time = get_time();
    }
    void record_backend_failure(const std::string& backend) {
        std::lock_guard<std::mutex> lk(state_mtx);
        auto& st = backend_stats[backend];
        st.beta = st.beta * 0.95 + 1.0;
        st.total_calls++;
        st.ewma_latency = std::max(st.ewma_latency * 1.5, 5.0);
    }
    bool is_on_cooldown(const std::string& backend) {
        std::lock_guard<std::mutex> lk(state_mtx);
        auto it = cooldown_until.find(backend);
        if (it == cooldown_until.end()) return false;
        if (get_time() >= it->second) {
            cooldown_until.erase(it);
            return false;
        }
        return true;
    }
    std::vector<std::string> get_backend_order() {
        std::lock_guard<std::mutex> lk(state_mtx);
        std::vector<std::pair<double, std::string>> scored;
        double now = get_time();
        for (auto& [name, client] : clients) {
            auto it = cooldown_until.find(name);
            if (it != cooldown_until.end() && now < it->second) continue;
            auto& st = backend_stats[name];
            double a = std::max(0.1, st.alpha);
            double b = std::max(0.1, st.beta);
            std::gamma_distribution<double> gamma_a(a, 1.0);
            std::gamma_distribution<double> gamma_b(b, 1.0);
            double x = gamma_a(rng_);
            double y = gamma_b(rng_);
            double sampled_rate = x / (x + y);
            double latency_factor = std::max(0.05, st.ewma_latency);
            double qos_score = sampled_rate / latency_factor;
            scored.push_back({qos_score, name});
        }
        std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
            return a.first > b.first;
        });
        std::vector<std::string> order;
        for (auto& [score, name] : scored) order.push_back(name);
        return order;
    }
    std::vector<std::string> get_active_models(const std::string& backend) {
        std::lock_guard<std::mutex> lk(state_mtx);
        auto it = active_models_.find(backend);
        if (it != active_models_.end()) {
            return it->second;
        }
        return {};
    }
    std::vector<std::string> get_active_models_sorted(const std::string& backend) {
        std::vector<std::string> models = get_active_models(backend);
        std::vector<std::pair<double, std::string>> scored;
        scored.reserve(models.size());
        for (const auto& m : models) {
            double ewma = get_model_ewma_latency(backend, m);
            scored.push_back({ewma, m});
        }
        std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        std::vector<std::string> result;
        result.reserve(scored.size());
        for (auto& [lat, name] : scored) result.push_back(name);
        return result;
    }
    void prune_model(const std::string& backend, const std::string& model) {
        std::lock_guard<std::mutex> lk(state_mtx);
        auto it = active_models_.find(backend);
        if (it != active_models_.end()) {
            auto& vec = it->second;
            vec.erase(std::remove(vec.begin(), vec.end(), model), vec.end());
        }
    }
    static bool is_model_not_found_error(ErrorCategory category, const std::string& error) {
        if (category != ErrorCategory::PERMANENT) return false;
        std::string low = error;
        std::transform(low.begin(), low.end(), low.begin(), ::tolower);
        return (low.find("model_not_found") != std::string::npos ||
                low.find("does not exist") != std::string::npos ||
                low.find("404") != std::string::npos ||
                low.find("not found") != std::string::npos ||
                low.find("unavailable model") != std::string::npos);
    }
};

class SelfModificationEngine {
public:
    explicit SelfModificationEngine(LLMBackendManager& llm);
    json proposeImprovement(const std::vector<std::string>& source_files);
private:
    LLMBackendManager& llm_;
    std::string extractCodeStructure(const std::string& source);
    std::string readFile(const std::string& path);
};

SelfModificationEngine::SelfModificationEngine(LLMBackendManager& llm) : llm_(llm) {}

std::string SelfModificationEngine::readFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return "";
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

std::string SelfModificationEngine::extractCodeStructure(const std::string& source) {
    std::string result;
    std::vector<std::pair<std::string, std::string>> scope_stack;
    std::vector<int> scope_brace_depth;
    int brace_depth = 0;
    int paren_depth = 0;
    std::string current_decl;
    enum State { CODE, LINE_COMMENT, BLOCK_COMMENT, STRING, CHAR } state = CODE;
    int backslash_count = 0;
    auto trim = [](std::string& s) {
        size_t start = s.find_first_not_of(" \t\n\r");
        if (start == std::string::npos) {
            s = "";
            return;
        }
        size_t end = s.find_last_not_of(" \t\n\r");
        s = s.substr(start, end - start + 1);
    };
    auto process_block = [&](const std::string& decl) {
        std::string d = decl;
        trim(d);
        if (d.empty()) return;
        std::string flat;
        for (char c : d) {
            if (c == '\n' || c == '\r' || c == '\t')
                flat += ' ';
            else
                flat += c;
        }
        trim(flat);
        if (flat.empty()) return;
        std::string indent(scope_stack.size() * 2, ' ');
        size_t ns_pos = flat.find("namespace ");
        if (ns_pos != std::string::npos &&
            (ns_pos == 0 || !std::isalnum(static_cast<unsigned char>(flat[ns_pos - 1])))) {
            std::string name = flat.substr(ns_pos + 10);
            trim(name);
            size_t end_pos = name.find_first_of(" {;");
            if (end_pos != std::string::npos) name = name.substr(0, end_pos);
            trim(name);
            if (!name.empty()) {
                result += indent + "namespace " + name + " {\n";
                scope_stack.push_back({"namespace", name});
                scope_brace_depth.push_back(brace_depth + 1);
                return;
            }
        }
        bool is_class = flat.find("class ") != std::string::npos;
        bool is_struct = flat.find("struct ") != std::string::npos;
        if (is_class || is_struct) {
            size_t pos = flat.find(is_class ? "class " : "struct ");
            std::string name = flat.substr(pos + (is_class ? 6 : 7));
            trim(name);
            size_t end_pos = name.find_first_of(" {;:");
            if (end_pos != std::string::npos) name = name.substr(0, end_pos);
            trim(name);
            if (!name.empty() && name.find(" ") == std::string::npos &&
                std::isalpha(static_cast<unsigned char>(name[0]))) {
                result += indent + (is_class ? "class " : "struct ") + name + " {\n";
                scope_stack.push_back({is_class ? "class" : "struct", name});
                scope_brace_depth.push_back(brace_depth + 1);
                return;
            }
        }
        if (paren_depth == 0 && flat.find("(") != std::string::npos &&
            flat.find(")") != std::string::npos) {
            size_t paren_start = flat.find("(");
            std::string prefix = flat.substr(0, paren_start);
            trim(prefix);
            size_t name_start = prefix.find_last_of(" \t:*&");
            std::string func_name =
                (name_start == std::string::npos) ? prefix : prefix.substr(name_start + 1);
            trim(func_name);
            if (!func_name.empty() && func_name.find(" ") == std::string::npos &&
                func_name != "if" && func_name != "for" && func_name != "while" &&
                func_name != "switch" && func_name != "catch" && func_name != "return") {
                result += indent + "func " + func_name + "(...)\n";
            }
        }
    };
    auto close_scopes = [&]() {
        while (!scope_stack.empty() && scope_brace_depth.back() > brace_depth) {
            std::string indent((scope_stack.size() - 1) * 2, ' ');
            result += indent + "}\n\n";
            scope_stack.pop_back();
            scope_brace_depth.pop_back();
        }
    };
    for (size_t i = 0; i < source.size(); ++i) {
        char c = source[i];
        char next = (i + 1 < source.size()) ? source[i + 1] : 0;
        if (state == CODE) {
            if (c == '/' && next == '/') {
                state = LINE_COMMENT;
                i++;
                continue;
            }
            if (c == '/' && next == '*') {
                state = BLOCK_COMMENT;
                i++;
                continue;
            }
            if (c == '"') {
                state = STRING;
                backslash_count = 0;
                current_decl += c;
                continue;
            }
            if (c == '\'') {
                state = CHAR;
                backslash_count = 0;
                current_decl += c;
                continue;
            }
            if (c == '(')
                paren_depth++;
            else if (c == ')')
                paren_depth--;
            if (c == '{') {
                if (paren_depth == 0) {
                    process_block(current_decl);
                    current_decl.clear();
                }
                brace_depth++;
            } else if (c == '}') {
                brace_depth--;
                close_scopes();
                current_decl.clear();
            } else if (c == ';' && paren_depth == 0) {
                current_decl.clear();
            } else {
                current_decl += c;
            }
        } else if (state == LINE_COMMENT) {
            if (c == '\n') state = CODE;
        } else if (state == BLOCK_COMMENT) {
            if (c == '*' && next == '/') {
                state = CODE;
                i++;
            }
        } else if (state == STRING || state == CHAR) {
            current_decl += c;
            if (c == '\\') {
                backslash_count++;
            } else {
                char quote = (state == STRING) ? '"' : '\'';
                if (c == quote && backslash_count % 2 == 0) state = CODE;
                backslash_count = 0;
            }
        }
    }
    close_scopes();
    return result;
}

json SelfModificationEngine::proposeImprovement(const std::vector<std::string>& source_files) {
    std::string full_source;
    for (const auto& file : source_files) {
        std::string content = readFile(file);
        if (!content.empty()) {
            full_source += "// --- FILE: " + file + " ---\n";
            full_source += content + "\n\n";
        }
    }
    if (full_source.empty()) {
        return {{"ok", false}, {"error", "Could not read any source files."}};
    }
    std::string structure = extractCodeStructure(full_source);
    if (structure.length() > 8000) {
        structure = structure.substr(0, 8000) + "\n... [TRUNCATED] ...\n";
    }
    std::string prompt =
        "You are the self-modification engine of a recursive C++ coding agent.\n"
        "Analyze the agent source structure below and propose ONE coherent improvement.\n"
        "Choose a target from: memory, planning, diagnosis, llm_orchestration, tool_executor, full_rewrite\n\n"
        "Output ONLY valid JSON matching this schema:\n"
        "{\n"
        "   \"target\": \"<subsystem>\",\n"
        "   \"proposal\": \"<clear description>\",\n"
        "   \"rationale\": \"<why this improves the agent>\",\n"
        "   \"diff_summary\": \"<what changes at high level>\",\n"
        "   \"confidence\": <0.0-1.0>\n"
        "}\n\n"
        "Agent source structure:\n" +
        structure;
    json schema = {
        {"type", "object"},
        {"properties",
         {{"target", {{"type", "string"}}},
          {"proposal", {{"type", "string"}}},
          {"rationale", {{"type", "string"}}},
          {"diff_summary", {{"type", "string"}}},
          {"confidence", {{"type", "number"}}}}},
        {"required", json::array({"target", "proposal", "rationale", "diff_summary", "confidence"})}};
    json result = llm_.structured_chat({{"user", prompt}}, schema, 0.3, 1024);
    if (!result.is_null() && result.contains("target")) {
        result["ok"] = true;
        return result;
    } else {
        return {{"ok", false}, {"error", "Failed to parse LLM response", "raw", result.dump()}};
    }
}

class CodeExecutor {
public:
    int max_execution_timeout = 30;
    int max_install_timeout = 60;
    size_t max_output_bytes = 131072;
    std::string last_compiled_exe;
    std::string run_shell_command(const std::string& command, int timeout = 30) {
        std::string full =
            "timeout " + std::to_string(timeout) + "s " + command +
            " 2>&1; echo RC:$?";
        FILE* pipe = popen(full.c_str(), "r");
        if (!pipe) {
            return "Error: cannot execute shell command";
        }
        std::string output;
        std::string tail_buffer;
        char buffer[512];
        bool truncated = false;
        while (fgets(buffer, sizeof(buffer), pipe)) {
            if (output.size() < max_output_bytes) {
                output += buffer;
            } else {
                if (!truncated) {
                    output +=
                        "\n...[OUTPUT TRUNCATED TO PREVENT MEMORY EXHAUSTION]...\n";
                    truncated = true;
                }
                tail_buffer += buffer;
                if (tail_buffer.size() > 1024) {
                    tail_buffer = tail_buffer.substr(tail_buffer.size() - 512);
                }
            }
        }
        pclose(pipe);
        if (truncated) {
            output += tail_buffer;
        }
        return "Shell output:\n" + output;
    }
    std::string compile_cpp(const std::string& source_file) {
        std::ifstream f(source_file);
        if (!f.is_open()) {
            return "COMPILATION_FAILED: cannot open source file: " + source_file;
        }
        std::string content(
            (std::istreambuf_iterator<char>(f)),
            std::istreambuf_iterator<char>()
        );
        std::string source_key;
        try {
            source_key = fs::absolute(source_file).lexically_normal().string();
        } catch (...) {
            source_key = source_file;
        }
        std::string content_hash = md5_hash(content);
        auto it = cpp_compile_cache_.find(source_key);
        if (it != cpp_compile_cache_.end() &&
            it->second.source_hash == content_hash &&
            fs::exists(it->second.exe_path)) {
            it->second.last_used = get_time();
            last_compiled_exe = it->second.exe_path;
            return last_compiled_exe;
        }
        std::string compile_flags = "-std=c++20 -O2";
        std::string link_flags;
        if (content.find("std::thread") != std::string::npos ||
            content.find("<thread>") != std::string::npos ||
            content.find("<mutex>") != std::string::npos ||
            content.find("<condition_variable>") != std::string::npos) {
            link_flags += " -pthread";
        }
        if (content.find("<curl/curl.h>") != std::string::npos) {
            link_flags += " -lcurl";
        }
        if (content.find("<openssl/") != std::string::npos) {
            link_flags += " -lssl -lcrypto";
        }
        if (content.find("<nlohmann/json.hpp>") != std::string::npos) {
            compile_flags += " -I/usr/include/nlohmann";
        }
        std::string cache_key =
            source_key + "|" + content_hash + "|" + compile_flags + "|" + link_flags;
        std::string out = "/tmp/agent_" + md5_hash(cache_key) + ".out";
        std::string old_exe;
        if (it != cpp_compile_cache_.end()) {
            old_exe = it->second.exe_path;
        }
        std::string cmd = "g++ " + compile_flags + " \"" + source_file +
                          "\" -o \"" + out + "\"" + link_flags;
        std::string output = run_shell_command(cmd, 15);
        if (output.rfind("RC:0") == std::string::npos || !fs::exists(out)) {
            if (fs::exists(out)) {
                fs::remove(out);
            }
            return "COMPILATION_FAILED: " + output;
        }
        last_compiled_exe = out;
        if (it != cpp_compile_cache_.end()) {
            it->second.source_hash = content_hash;
            it->second.exe_path = out;
            it->second.last_used = get_time();
        } else {
            if (cpp_compile_cache_.size() >= max_cpp_compile_cache_) {
                evict_oldest_compile_entry();
            }
            CppCompileCacheEntry entry;
            entry.source_key = source_key;
            entry.source_hash = content_hash;
            entry.exe_path = out;
            entry.last_used = get_time();
            cpp_compile_cache_[source_key] = entry;
        }
        if (!old_exe.empty() && old_exe != out && fs::exists(old_exe)) {
            fs::remove(old_exe);
        }
        return out;
    }
    std::string execute_file(
        const std::string& file_path,
        const std::string& stdin_input = "",
        int timeout_override = -1
    ) {
        fs::path path(file_path);
        std::string ext = path.extension().string();
        std::string cmd;
        if (
            ext.empty() &&
            !last_compiled_exe.empty() &&
            fs::exists(last_compiled_exe)
        ) {
            cmd = "\"" + last_compiled_exe + "\"";
        } else if (ext == ".py") {
            cmd = "python3 \"" + file_path + "\"";
        } else if (ext == ".cpp" || ext == ".cc" || ext == ".cxx") {
            std::string exe = compile_cpp(file_path);
            if (exe.find("COMPILATION_FAILED:") != std::string::npos) {
                return exe;
            }
            cmd = "\"" + exe + "\"";
        } else if (fs::exists(path) && access(path.c_str(), X_OK) == 0) {
            cmd = "\"" + path.string() + "\"";
        } else {
            return "Unsupported file type: " + ext;
        }
        const int effective_timeout =
            (timeout_override > 0) ? timeout_override : max_execution_timeout;
        std::string stdin_file;
        if (!stdin_input.empty()) {
            stdin_file =
                "/tmp/agent_stdin_" + std::to_string(rand()) + ".txt";
            std::ofstream f(stdin_file);
            f << stdin_input;
            f.close();
        }
        std::string full;
        if (!stdin_file.empty()) {
            full =
                "timeout " + std::to_string(effective_timeout) + "s " +
                cmd + " < \"" + stdin_file + "\" 2>&1; echo RC:$?";
        } else {
            full =
                "timeout " + std::to_string(effective_timeout) + "s " +
                cmd + " 2>&1; echo RC:$?";
        }
        FILE* pipe = popen(full.c_str(), "r");
        if (!pipe) {
            if (!stdin_file.empty()) fs::remove(stdin_file);
            return "Error: cannot execute file";
        }
        std::string output;
        std::string tail_buffer;
        char buffer[512];
        bool truncated = false;
        while (fgets(buffer, sizeof(buffer), pipe)) {
            if (output.size() < max_output_bytes) {
                output += buffer;
            } else {
                if (!truncated) {
                    output +=
                        "\n...[OUTPUT TRUNCATED TO PREVENT MEMORY EXHAUSTION]...\n";
                    truncated = true;
                }
                tail_buffer += buffer;
                if (tail_buffer.size() > 1024) {
                    tail_buffer = tail_buffer.substr(tail_buffer.size() - 512);
                }
            }
        }
        pclose(pipe);
        if (!stdin_file.empty()) {
            fs::remove(stdin_file);
        }
        if (truncated) {
            output += tail_buffer;
        }
        int rc = -1;
        size_t pos = output.rfind("RC:");
        if (pos != std::string::npos) {
            std::string rc_str = output.substr(pos + 3);
            rc = std::atoi(rc_str.c_str());
            output = output.substr(0, pos);
            if (!output.empty() && output.back() == '\n') output.pop_back();
        }
        std::ostringstream oss;
        oss << "Return code: " << rc << "\nSTDOUT:\n" << output;
        return oss.str();
    }
    std::string execute_code(
        const std::string& code,
        const std::string& language = "auto"
    ) {
        std::string lang = language;
        if (lang == "auto") {
            lang =
                (code.find("#include") != std::string::npos ||
                 code.find("int main(") != std::string::npos)
                    ? "cpp"
                    : "python";
        }
        std::string tmp =
            "/tmp/agent_code_" + std::to_string(rand()) +
            ((lang == "cpp") ? ".cpp" : ".py");
        std::ofstream f(tmp);
        f << code;
        f.close();
        std::string result = execute_file(tmp);
        fs::remove(tmp);
        return result;
    }
    std::string validate_syntax(
        const std::string& code,
        const std::string& language = "auto"
    ) {
        std::string lang = language;
        if (lang == "auto") {
            lang =
                (code.find("#include") != std::string::npos ||
                 code.find("int main(") != std::string::npos)
                    ? "cpp"
                    : "python";
        }
        if (lang == "cpp") {
            std::string tmp =
                "/tmp/agent_syntax_" + std::to_string(rand()) + ".cpp";
            std::ofstream f(tmp);
            f << code;
            f.close();
            std::string cmd = "g++ -std=c++20 -fsyntax-only \"" + tmp + "\"";
            std::string out = run_shell_command(cmd, 10);
            fs::remove(tmp);
            if (out.rfind("RC:0") != std::string::npos) {
                return "Syntax validation PASSED.";
            }
            return "Syntax validation FAILED: " + out;
        }
        std::string tmp =
            "/tmp/agent_syntax_" + std::to_string(rand()) + ".py";
        std::ofstream f(tmp);
        f << code;
        f.close();
        std::string cmd = "python3 -m py_compile \"" + tmp + "\"";
        std::string out = run_shell_command(cmd, 10);
        fs::remove(tmp);
        if (out.rfind("RC:0") != std::string::npos) {
            return "Syntax validation PASSED.";
        }
        return "Syntax validation FAILED: " + out;
    }

private:
    struct CppCompileCacheEntry {
        std::string source_key;
        std::string source_hash;
        std::string exe_path;
        double last_used = 0.0;
    };
    static constexpr size_t max_cpp_compile_cache_ = 12;
    std::unordered_map<std::string, CppCompileCacheEntry> cpp_compile_cache_;
    void evict_oldest_compile_entry() {
        if (cpp_compile_cache_.empty()) {
            return;
        }
        auto oldest = cpp_compile_cache_.begin();
        for (auto it = cpp_compile_cache_.begin();
             it != cpp_compile_cache_.end();
             ++it) {
            if (it->second.last_used < oldest->second.last_used) {
                oldest = it;
            }
        }
        if (fs::exists(oldest->second.exe_path)) {
            fs::remove(oldest->second.exe_path);
        }
        cpp_compile_cache_.erase(oldest);
    }
};

struct MemoryEntry {
    std::string task_hash;
    std::string task_text;
    std::string solution_code;
    std::unordered_map<std::string, double> raw_tf;
    double norm = 0.0;
    double success_timestamp = 0.0;
    int api_calls_saved = 0;
    int access_count = 0;
    double last_access_time = 0.0;
    int term_count = 0;
    bool is_plan = false;
    int success_count = 0;
    int failure_count = 0;
    uint32_t doc_id = 0;
    double reward = 0.0;
    int episode_count = 0;
    std::unordered_map<std::string, std::string> file_artifacts;
};

class SemanticMemoryManager {
public:
    explicit SemanticMemoryManager(const std::string& db_path = "./agent_memory_db")
        : db_path(db_path) {
        fs::create_directories(db_path);
        load_all();
    }
    bool retrieve_replayable(
        const std::string& query,
        const std::vector<std::string>& required_files,
        const std::vector<json>& checks,
        std::unordered_map<std::string, std::string>& out_artifacts,
        double* out_reward = nullptr) {
        out_artifacts.clear();
        if (required_files.empty()) return false;
        std::string task_hash = md5_hash(query);
        std::shared_ptr<MemoryEntry> entry;
        if (auto it = exact.find(task_hash); it != exact.end()) {
            entry = it->second;
            stats.exact_hits++;
            stats.retrieval_count++;
            if (entry->success_count + entry->failure_count > 0) {
                last_retrieval[task_hash] = {1.0, 0.0, entry->task_hash};
            }
            entry->access_count++;
            entry->last_access_time = get_time();
            entry->api_calls_saved++;
        } else {
            entry = best_semantic_entry(query, 0.35, nullptr);
            if (entry) {
                stats.semantic_hits++;
                stats.retrieval_count++;
                entry->access_count++;
                entry->last_access_time = get_time();
                entry->api_calls_saved++;
                last_retrieval[task_hash] = {1.0, 0.0, entry->task_hash};
            } else {
                stats.misses++;
                return false;
            }
        }
        if (!entry) return false;
        bool covered = true;
        for (const auto& f : required_files) {
            if (!entry->file_artifacts.count(f)) { covered = false; break; }
        }
        if (!covered) return false;
        for (const auto& c : checks) {
            if (c.value("type", "") != "execution") continue;
            const std::string fp = c.value("file", "");
            auto it = entry->file_artifacts.find(fp);
            if (it == entry->file_artifacts.end()) return false;
            const std::string expect = c.value("expect", "");
            if (expect.empty()) continue;
            std::string probe = it->second;
            std::string lower_probe = probe;
            std::string lower_expect = expect;
            std::transform(lower_probe.begin(), lower_probe.end(), lower_probe.begin(), ::tolower);
            std::transform(lower_expect.begin(), lower_expect.end(), lower_expect.begin(), ::tolower);
            if (lower_probe.find(lower_expect) == std::string::npos) {
                return false;
            }
        }
        out_artifacts = entry->file_artifacts;
        if (out_reward) *out_reward = entry->reward;
        return true;
    }
    std::string retrieve_similar(
        const std::string& query,
        double threshold = 0.35,
        double* out_reward = nullptr
    ) {
        std::string task_hash = md5_hash(query);
        if (auto it = exact.find(task_hash); it != exact.end()) {
            auto entry = it->second;
            double now = get_time();
            entry->access_count++;
            entry->last_access_time = now;
            entry->api_calls_saved++;
            last_retrieval[task_hash] = {1.0, 0.0, entry->task_hash};
            stats.exact_hits++;
            stats.retrieval_count++;
            save_if_dirty();
            if (out_reward) *out_reward = entry->reward;
            return entry->solution_code;
        }
        auto entry = best_semantic_entry(query, threshold, out_reward);
        if (entry) {
            stats.semantic_hits++;
            stats.retrieval_count++;
            entry->access_count++;
            entry->last_access_time = get_time();
            entry->api_calls_saved++;
            last_retrieval[task_hash] = {1.0, 0.0, entry->task_hash};
            save_if_dirty();
            if (out_reward) *out_reward = entry->reward;
            return entry->solution_code;
        }
        stats.misses++;
        return "";
    }
    void store(
        const std::string& task,
        const std::string& solution,
        double reward = 0.0,
        const std::unordered_map<std::string, std::string>& file_artifacts = {}
    ) {
        std::string safe_task = sanitize_utf8(task);
        std::string safe_solution = sanitize_utf8(solution);
        std::string task_hash = md5_hash(safe_task);
        bool is_plan = false;
        try {
            json parsed = json::parse(safe_solution);
            if (parsed.is_array()) {
                is_plan = true;
                for (auto& action : parsed) {
                    if (!action.contains("tool")) { is_plan = false; break; }
                }
            }
        } catch (...) {}
        std::shared_ptr<MemoryEntry> entry;
        bool is_new = false;
        if (auto it = exact.find(task_hash); it != exact.end()) {
            entry = it->second;
            bool task_text_changed = (entry->task_text != safe_task);
            entry->solution_code = safe_solution;
            entry->reward = (entry->reward * entry->episode_count + reward) / (entry->episode_count + 1);
            entry->episode_count++;
            if (task_text_changed) {
                entry->task_text = safe_task;
                remove_doc_from_index(entry);
                compute_raw_tf(entry);
                add_doc_to_index(entry);
            }
        } else {
            entry = std::make_shared<MemoryEntry>();
            entry->task_hash = task_hash;
            entry->task_text = safe_task;
            entry->solution_code = safe_solution;
            entry->reward = reward;
            entry->episode_count = 1;
            compute_raw_tf(entry);
            if (static_cast<int>(memory.size()) >= max_entries) {
                evict_lowest_scoring_entry();
            }
            entry->doc_id = static_cast<uint32_t>(memory.size());
            memory.push_back(entry);
            exact[task_hash] = entry;
            add_doc_to_index(entry);
            is_new = true;
        }
        entry->success_timestamp = get_time();
        entry->is_plan = is_plan;
        entry->access_count++;
        if (!file_artifacts.empty()) {
            for (const auto& [fp, content] : file_artifacts) {
                entry->file_artifacts[fp] = sanitize_utf8(content);
            }
        }
        if (is_new) {
            recompute_global_idf();
        }
        memory_dirty_ = true;
        save_if_dirty();
    }
    void store_failure(
        const std::string& task_hash,
        const std::string& error_type,
        const std::string& error_output,
        const std::vector<std::string>& fix_actions,
        const std::string& hint
    ) {
        std::string sig = error_output.substr(0, 200);
        std::replace(sig.begin(), sig.end(), '\n', ' ');
        failure_patterns.push_back({
            task_hash, error_type, sig, hint, fix_actions, 1, get_time()
        });
        failures_dirty_ = true;
        save_if_dirty();
    }
    std::string get_hint_for_error(const std::string& error_output) {
        std::string etype = error_type_from_output(error_output);
        std::string sig = error_output.substr(0, 200);
        std::replace(sig.begin(), sig.end(), '\n', ' ');
        for (auto& pattern : failure_patterns) {
            bool same_type = pattern.error_type == etype;
            bool signature_matches =
                pattern.error_signature.find(sig) != std::string::npos ||
                sig.find(pattern.error_signature) != std::string::npos;
            if (same_type && signature_matches) {
                pattern.occurrence_count++;
                pattern.last_seen = get_time();
                failures_dirty_ = true;
                save_if_dirty();
                return pattern.hint;
            }
        }
        return "";
    }
    void report_outcome(
        const std::string& task_hash,
        bool success
    ) {
        auto it = last_retrieval.find(task_hash);
        if (it != last_retrieval.end()) {
            auto [cos, jac, entry_hash] = it->second;
            if (auto entry_it = exact.find(entry_hash); entry_it != exact.end()) {
                auto entry = entry_it->second;
                if (success) entry->success_count++;
                else entry->failure_count++;
                memory_dirty_ = true;
            }
            last_retrieval.erase(it);
        }
        double thresh = 0.35;
        if (success) {
            thresh = std::max(0.20, thresh * 0.95);
        } else {
            consec_fail[task_hash]++;
            if (consec_fail[task_hash] >= 2) {
                thresh = std::min(0.50, thresh * 1.05);
                consec_fail[task_hash] = 0;
            }
        }
        dynamic_thresholds[task_hash] = thresh;
        save_if_dirty();
    }
    void print_stats() const {
        int saved = 0;
        for (const auto& entry : memory) saved += entry->api_calls_saved;
        std::cout << "Memory entries: " << memory.size() << "/" << max_entries << "\n";
        std::cout << "Hits exact=" << stats.exact_hits
                  << " semantic=" << stats.semantic_hits
                  << " misses=" << stats.misses << "\n";
        std::cout << "API calls saved: " << saved << "\n";
    }

private:
    struct CompactPosting { uint32_t doc_id; double tf; };
    struct InvertedList { std::vector<CompactPosting> postings; double max_tf_idf = 0.0; };
    struct FailurePattern {
        std::string task_hash, error_type, error_signature, hint;
        std::vector<std::string> fix_actions;
        int occurrence_count = 1; double last_seen = 0.0;
    };
    struct Stats { int exact_hits=0, semantic_hits=0, misses=0, retrieval_count=0; };
    std::vector<std::shared_ptr<MemoryEntry>> memory;
    std::unordered_map<std::string, std::shared_ptr<MemoryEntry>> exact;
    std::unordered_map<std::string, InvertedList> inverted_index_;
    std::unordered_map<std::string, double> idf;
    std::unordered_map<std::string, int> doc_freq_;
    int total_docs = 0; double avgdl_ = 0.0;
    Stats stats;
    std::unordered_map<std::string, double> dynamic_thresholds;
    int max_entries = 50; std::string db_path;
    std::vector<FailurePattern> failure_patterns;
    std::unordered_map<std::string, std::tuple<double, double, std::string>> last_retrieval;
    double weights[3] = {3.0, 2.0, -1.5}; double learning_rate = 0.02;
    std::unordered_map<std::string, int> consec_fail;
    std::vector<double> accum_dot_; std::vector<int> accum_overlap_;
    std::vector<uint32_t> doc_eval_gen_; std::vector<double> min_dot_req_;
    std::vector<uint32_t> active_candidates_; uint32_t current_eval_gen_ = 0;
    bool memory_dirty_ = false;
    bool failures_dirty_ = false;
    bool index_dirty_ = false;
    static std::string sanitize_utf8(const std::string& input) {
        std::string result;
        result.reserve(input.size());
        size_t i = 0;
        while (i < input.size()) {
            unsigned char c = input[i];
            size_t len = 1;
            if (c < 0x80) len = 1;
            else if ((c >> 5) == 0x06) len = 2;
            else if ((c >> 4) == 0x0E) len = 3;
            else if ((c >> 3) == 0x1E) len = 4;
            else { result += '?'; i++; continue; }
            if (i + len > input.size()) break;
            bool valid = true;
            for (size_t j = 1; j < len; ++j) {
                if ((input[i + j] & 0xC0) != 0x80) { valid = false; break; }
            }
            if (valid) result.append(input, i, len);
            else result += '?';
            i += len;
        }
        return result;
    }
    void ensure_accumulator_capacity(size_t n) {
        if (accum_dot_.size() < n) {
            accum_dot_.resize(n, 0.0);
            accum_overlap_.resize(n, 0);
            doc_eval_gen_.resize(n, 0);
            min_dot_req_.resize(n, 0.0);
        }
    }
    std::shared_ptr<MemoryEntry> best_semantic_entry(
        const std::string& query,
        double threshold,
        double* out_reward
    ) {
        if (memory.empty()) return nullptr;
        auto query_tokens = tokenize(query);
        if (query_tokens.empty()) return nullptr;
        std::unordered_map<std::string, int> query_freq;
        for (const auto& t : query_tokens) query_freq[t]++;
        std::string task_hash = md5_hash(query);
        double now = get_time();
        double eff_threshold = threshold;
        if (auto it = dynamic_thresholds.find(task_hash); it != dynamic_thresholds.end()) {
            eff_threshold = it->second;
        }
        const double k1 = 1.5;
        const double b = 0.75;
        ensure_accumulator_capacity(memory.size());
        current_eval_gen_++;
        active_candidates_.clear();
        for (const auto& [term, q_freq] : query_freq) {
            auto inv_it = inverted_index_.find(term);
            if (inv_it == inverted_index_.end()) continue;
            double idf_val = idf.count(term) ? idf.at(term) : 0.0;
            if (idf_val <= 0.0) continue;
            const auto& postings = inv_it->second.postings;
            for (const auto& posting : postings) {
                uint32_t doc_id = posting.doc_id;
                if (doc_id >= memory.size()) continue;
                if (doc_eval_gen_[doc_id] != current_eval_gen_) {
                    doc_eval_gen_[doc_id] = current_eval_gen_;
                    accum_dot_[doc_id] = 0.0;
                    active_candidates_.push_back(doc_id);
                }
                double f = posting.tf;
                double dl = memory[doc_id]->term_count;
                double dl_ratio = (avgdl_ > 0) ? (dl / avgdl_) : 1.0;
                double numerator = f * (k1 + 1.0);
                double denominator = f + k1 * (1.0 - b + b * dl_ratio);
                accum_dot_[doc_id] += idf_val * (numerator / denominator);
            }
        }
        std::shared_ptr<MemoryEntry> best_entry;
        double best_score = 0.0;
        for (uint32_t doc_id : active_candidates_) {
            if (accum_dot_[doc_id] <= 0.0) continue;
            auto& entry = memory[doc_id];
            double bm25_score = accum_dot_[doc_id];
            double age_days = (now - entry->success_timestamp) / 86400.0;
            double freshness = 1.0 - 0.1 * std::min(1.0, std::max(0.0, (age_days - 1.0) / 6.0));
            double reliability = memory_reliability(entry);
            double normalized_bm25 = bm25_score / (bm25_score + 2.0);
            double score = (0.7 * normalized_bm25 + 0.3 * reliability) * freshness;
            if (score > best_score) {
                best_score = score;
                best_entry = entry;
            }
        }
        if (best_entry && best_score >= eff_threshold) {
            if (out_reward) *out_reward = best_entry->reward;
            return best_entry;
        }
        return nullptr;
    }
    void load_all() {
        load_memory();
        load_failures();
    }
    void load_memory() {
        fs::path mem_file = fs::path(db_path) / "semantic_memory.json";
        if (!fs::exists(mem_file)) return;
        try {
            std::ifstream f(mem_file);
            std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            content = sanitize_utf8(content);
            json data = json::parse(content, nullptr, false);
            if (data.is_discarded()) return;
            memory.clear(); exact.clear();
            for (auto& item : data) {
                auto entry = std::make_shared<MemoryEntry>();
                entry->task_hash = item.value("task_hash", "");
                entry->task_text = item.value("task_text", "");
                entry->solution_code = item.value("solution_code", "");
                entry->success_timestamp = item.value("timestamp", 0.0);
                entry->api_calls_saved = item.value("api_calls_saved", 0);
                entry->access_count = item.value("access_count", 0);
                entry->last_access_time = item.value("last_access_time", 0.0);
                entry->is_plan = item.value("is_plan", false);
                entry->success_count = item.value("success_count", 0);
                entry->failure_count = item.value("failure_count", 0);
                entry->reward = item.value("reward", 0.0);
                entry->episode_count = item.value("episode_count", 0);
                entry->doc_id = static_cast<uint32_t>(memory.size());
                if (item.contains("file_artifacts") && item["file_artifacts"].is_object()) {
                    for (auto& [k, v] : item["file_artifacts"].items()) {
                        if (v.is_string()) entry->file_artifacts[k] = v.get<std::string>();
                    }
                }
                compute_raw_tf(entry);
                memory.push_back(entry);
                exact[entry->task_hash] = entry;
            }
            rebuild_index();
        } catch (const std::exception& e) {
            std::cerr << "[Memory] Load error: " << e.what() << "\n";
        }
    }
    void load_failures() {
        fs::path fail_file = fs::path(db_path) / "failure_patterns.json";
        if (!fs::exists(fail_file)) return;
        try {
            std::ifstream f(fail_file);
            std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            content = sanitize_utf8(content);
            json data = json::parse(content, nullptr, false);
            if (data.is_discarded()) return;
            for (auto& item : data) {
                FailurePattern pattern;
                pattern.task_hash = item.value("task_hash", "");
                pattern.error_type = item.value("error_type", "");
                pattern.error_signature = item.value("error_signature", "");
                if (item.contains("fix_actions")) {
                    for (auto& action : item["fix_actions"]) {
                        pattern.fix_actions.push_back(action);
                    }
                }
                pattern.hint = item.value("hint", "");
                pattern.occurrence_count = item.value("occurrence_count", 1);
                pattern.last_seen = item.value("last_seen", 0.0);
                failure_patterns.push_back(pattern);
            }
        } catch (...) {}
    }
    void save_if_dirty() {
        if (memory_dirty_) save_memory();
        if (failures_dirty_) save_failures();
    }
    void save_memory() {
        json data = json::array();
        for (const auto& entry : memory) {
            json artifacts = json::object();
            for (const auto& [k, v] : entry->file_artifacts) {
                artifacts[k] = v;
            }
            data.push_back({
                {"task_hash", entry->task_hash},
                {"task_text", entry->task_text},
                {"solution_code", entry->solution_code},
                {"timestamp", entry->success_timestamp},
                {"api_calls_saved", entry->api_calls_saved},
                {"access_count", entry->access_count},
                {"last_access_time", entry->last_access_time},
                {"is_plan", entry->is_plan},
                {"success_count", entry->success_count},
                {"failure_count", entry->failure_count},
                {"reward", entry->reward},
                {"episode_count", entry->episode_count},
                {"file_artifacts", artifacts}
            });
        }
        std::ofstream(fs::path(db_path) / "semantic_memory.json") << data.dump(2);
        memory_dirty_ = false;
    }
    void save_failures() {
        json data = json::array();
        for (const auto& pattern : failure_patterns) {
            json fixes = json::array();
            for (const auto& fix : pattern.fix_actions) fixes.push_back(fix);
            data.push_back({
                {"task_hash", pattern.task_hash},
                {"error_type", pattern.error_type},
                {"error_signature", pattern.error_signature},
                {"fix_actions", fixes},
                {"hint", pattern.hint},
                {"occurrence_count", pattern.occurrence_count},
                {"last_seen", pattern.last_seen}
            });
        }
        std::ofstream(fs::path(db_path) / "failure_patterns.json") << data.dump(2);
        failures_dirty_ = false;
    }
    void compute_raw_tf(std::shared_ptr<MemoryEntry> entry) {
        entry->raw_tf.clear();
        auto tokens = tokenize(entry->task_text);
        if (tokens.empty()) {
            entry->term_count = 0;
            entry->norm = 0.0;
            return;
        }
        std::unordered_map<std::string, int> freq;
        for (const auto& token : tokens) freq[token]++;
        int max_freq = 0;
        for (const auto& [_, count] : freq) max_freq = std::max(max_freq, count);
        for (const auto& [term, count] : freq) {
            entry->raw_tf[term] = static_cast<double>(count) / max_freq;
        }
        entry->term_count = static_cast<int>(entry->raw_tf.size());
    }
    void add_doc_to_index(const std::shared_ptr<MemoryEntry>& entry) {
        for (const auto& [term, raw_tf] : entry->raw_tf) {
            auto& inv = inverted_index_[term];
            inv.postings.push_back({entry->doc_id, raw_tf});
            doc_freq_[term]++;
        }
    }
    void remove_doc_from_index(const std::shared_ptr<MemoryEntry>& entry) {
        for (const auto& [term, _] : entry->raw_tf) {
            auto inv_it = inverted_index_.find(term);
            if (inv_it == inverted_index_.end()) continue;
            auto& vec = inv_it->second.postings;
            vec.erase(std::remove_if(vec.begin(), vec.end(),
                [&](const CompactPosting& p) { return p.doc_id == entry->doc_id; }),
                vec.end());
            if (auto df_it = doc_freq_.find(term); df_it != doc_freq_.end()) {
                if (--df_it->second <= 0) doc_freq_.erase(df_it);
            }
            if (vec.empty()) inverted_index_.erase(inv_it);
        }
    }
    void rebuild_index() {
        total_docs = static_cast<int>(memory.size());
        doc_freq_.clear();
        idf.clear();
        inverted_index_.clear();
        double total_terms = 0.0;
        for (uint32_t i = 0; i < memory.size(); ++i) {
            memory[i]->doc_id = i;
            total_terms += memory[i]->term_count;
            for (const auto& [term, _] : memory[i]->raw_tf) {
                doc_freq_[term]++;
            }
        }
        avgdl_ = total_docs > 0 ? total_terms / total_docs : 1.0;
        recompute_global_idf();
        for (uint32_t i = 0; i < memory.size(); ++i) {
            auto& entry = memory[i];
            for (const auto& [term, raw_tf] : entry->raw_tf) {
                double idf_val = idf.count(term) ? idf[term] : 0.0;
                auto& inv = inverted_index_[term];
                inv.postings.push_back({i, raw_tf});
                inv.max_tf_idf = std::max(inv.max_tf_idf, raw_tf * idf_val);
            }
        }
        index_dirty_ = false;
    }
    void recompute_global_idf() {
        idf.clear();
        for (const auto& [term, df] : doc_freq_) {
            idf[term] = std::log(1.0 + (total_docs - df + 0.5) / (df + 0.5));
        }
    }
    void evict_lowest_scoring_entry() {
        if (memory.empty()) return;
        auto to_evict = std::min_element(
            memory.begin(), memory.end(),
            [&](const auto& a, const auto& b) {
                double score_a = 0.65 * std::min(a->access_count, 10) / 10.0 + 0.35 * memory_reliability(a);
                double score_b = 0.65 * std::min(b->access_count, 10) / 10.0 + 0.35 * memory_reliability(b);
                return score_a < score_b;
            }
        );
        if (to_evict != memory.end()) {
            remove_doc_from_index(*to_evict);
            exact.erase((*to_evict)->task_hash);
            memory.erase(to_evict);
            for (uint32_t i = 0; i < memory.size(); ++i) memory[i]->doc_id = i;
            total_docs = static_cast<int>(memory.size());
            index_dirty_ = true;
        }
    }
    double memory_reliability(const std::shared_ptr<MemoryEntry>& entry) const {
        int total = entry->success_count + entry->failure_count;
        if (total <= 0) return 0.5;
        return (entry->success_count + 1.0) / (total + 2.0);
    }
};

class FileManager {
public:
    explicit FileManager(const std::string& backup_dir) : backup_dir(backup_dir) {
        fs::create_directories(backup_dir);
    }
    std::string write_file(const std::string& file_path, const std::string& content) {
        fs::path path(file_path);
        if (path.has_parent_path()) {
            fs::create_directories(path.parent_path());
        }
        std::string fixed = content;
        fixed = std::regex_replace(
            fixed,
            std::regex(R"(#include\s*<json/json.h>)"),
            "#include <nlohmann/json.hpp>"
        );
        fs::path tmp = path.string() + ".tmp";
        {
            std::ofstream f(tmp);
            if (!f.is_open()) {
                return "Error: Cannot create temporary file for writing: " + tmp.string();
            }
            f << fixed;
        }
        try {
            fs::rename(tmp, path);
        } catch (const std::exception& e) {
            return "Error: Failed to rename temporary file for " + file_path +
                   ": " + std::string(e.what());
        }
        return "File written successfully: " + file_path;
    }
    std::string read_file(const std::string& file_path) const {
        fs::path path(file_path);
        if (!fs::exists(path)) {
            return "Error: File not found: " + file_path;
        }
        if (!fs::is_regular_file(path)) {
            return "Error: Not a regular file: " + file_path;
        }
        try {
            std::ifstream f(file_path, std::ios::binary);
            if (!f.is_open()) {
                return "Error: Cannot open file: " + file_path;
            }
            std::ostringstream ss;
            ss << f.rdbuf();
            if (f.bad()) {
                return "Error reading file: " + file_path;
            }
            std::string content = ss.str();
            if (content.empty()) {
                return "Error: File is empty: " + file_path;
            }
            return content;
        } catch (const std::exception& e) {
            return "Error reading file: " + std::string(e.what());
        }
    }
    std::string backup_file(const std::string& file_path) const {
        fs::path path(file_path);
        if (!fs::exists(path)) {
            return "";
        }
        const std::string base_name = path.filename().string();
        std::lock_guard<std::mutex> lock(version_mutex_);
        int next_ver = next_version_locked(base_name);
        fs::path backup = fs::path(backup_dir) /
                          (base_name + "_v" + std::to_string(next_ver) + ".bak");
        fs::copy_file(path, backup, fs::copy_options::overwrite_existing);
        return backup.string();
    }
    std::string restore_latest_backup(const std::string& file_path) const {
        fs::path path(file_path);
        const std::string base_name = path.filename().string();
        std::lock_guard<std::mutex> lock(version_mutex_);
        build_version_index_locked();
        auto it = next_version_.find(base_name);
        if (it == next_version_.end() || it->second <= 1) {
            return "";
        }
        int latest_ver = it->second - 1;
        fs::path backup = fs::path(backup_dir) / (base_name + "_v" + std::to_string(latest_ver) + ".bak");
        if (fs::exists(backup)) {
            fs::copy_file(backup, path, fs::copy_options::overwrite_existing);
            return "Restored " + file_path + " from backup v" + std::to_string(latest_ver);
        }
        return "";
    }

private:
    fs::path backup_dir;
    mutable std::mutex version_mutex_;
    mutable std::unordered_map<std::string, int> next_version_;
    mutable bool version_index_initialized_ = false;
    static bool extract_backup_base_version(const std::string& file_name,
                                            std::string& base,
                                            int& version) {
        const std::string suffix = ".bak";
        if (file_name.size() <= suffix.size() ||
            file_name.compare(file_name.size() - suffix.size(), suffix.size(), suffix) != 0) {
            return false;
        }
        const std::string core = file_name.substr(0, file_name.size() - suffix.size());
        const size_t sep = core.rfind("_v");
        if (sep == std::string::npos || sep + 2 >= core.size()) {
            return false;
        }
        const std::string version_text = core.substr(sep + 2);
        for (char c : version_text) {
            if (c < '0' || c > '9') {
                return false;
            }
        }
        base = core.substr(0, sep);
        version = std::atoi(version_text.c_str());
        return true;
    }
    void build_version_index_locked() const {
        if (version_index_initialized_) {
            return;
        }
        next_version_.clear();
        for (const auto& entry : fs::directory_iterator(backup_dir)) {
            const std::string name = entry.path().filename().string();
            std::string base;
            int version = 0;
            if (extract_backup_base_version(name, base, version)) {
                int& next = next_version_[base];
                next = std::max(next, version + 1);
            }
        }
        version_index_initialized_ = true;
    }
    int next_version_locked(const std::string& base_name) const {
        build_version_index_locked();
        auto it = next_version_.find(base_name);
        if (it == next_version_.end()) {
            next_version_[base_name] = 1;
            return 1;
        }
        return it->second++;
    }
};

class ToolExecutor {
public:
    ToolExecutor() {}
    CodeExecutor executor;
    FileManager file_manager{".agent_backups"};
    std::string execute_action_dict(json action) {
        try {
            if (!action.is_object()) {
                return "Error: Tool action is not a JSON object.";
            }
            action = apply_aliases(action);
            if (!action.contains("tool") || !action["tool"].is_string()) {
                return "Error: Tool action is missing a string 'tool' field.";
            }
            std::string tool = action["tool"].get<std::string>();
            json args = (action.contains("args") && action["args"].is_object())
                            ? action["args"]
                            : json::object();
            static const std::map<std::string, std::set<std::string>> required = {
                {"write_file", {"file_path", "content"}},
                {"read_file", {"file_path"}},
                {"execute_file", {"file_path"}},
                {"compile_cpp", {"file_path"}},
                {"run_shell_command", {"command"}},
                {"make_directory", {"dir_path"}},
                {"install_package", {"package"}},
                {"finish", {}}
            };
            if (!required.count(tool)) {
                return "Error: Tool '" + tool + "' is not recognized.";
            }
            std::set<std::string> missing;
            for (const auto& arg : required.at(tool)) {
                if (!args.contains(arg)) {
                    missing.insert(arg);
                }
            }
            if (!missing.empty()) {
                std::string msg;
                for (const auto& m : missing) {
                    msg += m + " ";
                }
                return "Error: Missing required arguments for '" + tool +
                       "': " + msg;
            }
            if (tool == "write_file") {
                std::string fp = get_string_arg(args, "file_path");
                std::string content = get_string_arg(args, "content");
                if (fp.empty()) {
                    return "Error: write_file 'file_path' must be a non-empty string.";
                }
                if (content.empty()) {
                    return "Error: write_file 'content' must be a non-empty string.";
                }
                std::string syntax = validate_content_syntax(fp, content);
                if (!syntax.empty() &&
                    syntax.find("FAILED") != std::string::npos) {
                    return "Error: " + syntax +
                           "\nFile was NOT written. Fix the syntax error and try again.";
                }
                return file_manager.write_file(fp, content);
            }
            if (tool == "read_file") {
                std::string fp = get_string_arg(args, "file_path");
                if (fp.empty()) {
                    return "Error: read_file 'file_path' must be a non-empty string.";
                }
                return file_manager.read_file(fp);
            }
            if (tool == "execute_file") {
                std::string fp = get_string_arg(args, "file_path");
                if (fp.empty()) {
                    return "Error: execute_file 'file_path' must be a non-empty string.";
                }
                std::string stdin_input = get_string_arg(args, "stdin");
                return executor.execute_file(fp, stdin_input);
            }
            if (tool == "make_directory") {
                std::string dir = get_string_arg(args, "dir_path");
                if (dir.empty()) {
                    return "Error: make_directory 'dir_path' must be a non-empty string.";
                }
                fs::create_directories(dir);
                return "Directory created: " + dir;
            }
            if (tool == "install_package") {
                std::string pkg = get_string_arg(args, "package");
                if (pkg.empty()) {
                    return "Error: install_package 'package' must be a non-empty string.";
                }
                return executor.run_shell_command(
                    "pip install " + pkg,
                    executor.max_install_timeout
                );
            }
            if (tool == "compile_cpp") {
                std::string fp = get_string_arg(args, "file_path");
                if (fp.empty()) {
                    return "Error: compile_cpp 'file_path' must be a non-empty string.";
                }
                return executor.compile_cpp(fp);
            }
            if (tool == "run_shell_command") {
                std::string cmd = get_string_arg(args, "command");
                if (cmd.empty()) {
                    return "Error: run_shell_command 'command' must be a non-empty string.";
                }
                return executor.run_shell_command(cmd);
            }
            if (tool == "finish") {
                return "Task marked as finished. Verifying results...";
            }
            return "Unknown tool";
        } catch (const std::exception& e) {
            return "Exception executing tool: " + std::string(e.what());
        }
    }
    std::vector<json> extract_actions(const std::string& text) {
        std::vector<json> objects;
        std::set<std::string> seen;
        auto add_object = [&](const json& obj) {
            if (obj.is_object() && obj.contains("tool")) {
                json normalized = apply_aliases(obj);
                std::string key = normalized.dump();
                if (!seen.count(key)) {
                    seen.insert(key);
                    objects.push_back(normalized);
                }
            }
        };
        size_t pos = 0;
        while ((pos = text.find("```", pos)) != std::string::npos) {
            size_t end_pos = text.find("```", pos + 3);
            if (end_pos == std::string::npos) {
                break;
            }
            std::string block = text.substr(pos + 3, end_pos - pos - 3);
            size_t start_idx = 0;
            while (start_idx < block.size() &&
                   std::isspace(static_cast<unsigned char>(block[start_idx]))) {
                start_idx++;
            }
            if (block.compare(start_idx, 4, "json") == 0) {
                start_idx += 4;
            }
            parse_and_collect_json(block.substr(start_idx), add_object);
            pos = end_pos + 3;
        }
        if (objects.empty()) {
            parse_and_collect_json(text, add_object);
        }
        if (objects.empty()) {
            json direct = extract_write_file_direct(text);
            if (!direct.is_null() && !direct.empty()) {
                add_object(direct);
            }
        }
        if (objects.empty()) {
            auto xml_actions = extract_xml_tool_calls(text);
            for (const auto& act : xml_actions) {
                add_object(act);
            }
        }
        return objects;
    }

private:
    mutable std::mutex syntax_validation_mutex_;
    std::unordered_map<std::string, std::string> syntax_validation_cache_;
    static std::string get_string_arg(
        const json& args,
        const std::string& key
    ) {
        if (!args.is_object() ||
            !args.contains(key) ||
            !args[key].is_string()) {
            return "";
        }
        return args[key].get<std::string>();
    }
    static json apply_aliases(json action) {
        if (!action.is_object()) {
            return action;
        }
        if ((!action.contains("tool") || !action["tool"].is_string()) &&
            action.contains("name") && action["name"].is_string()) {
            action["tool"] = action["name"];
            action.erase("name");
        }
        if (!action.contains("args") || !action["args"].is_object()) {
            action["args"] = json::object();
        }
        std::vector<std::string> to_move;
        for (auto it = action.begin(); it != action.end(); ++it) {
            if (it.key() != "tool" && it.key() != "args") {
                to_move.push_back(it.key());
            }
        }
        for (const auto& k : to_move) {
            action["args"][k] = action[k];
            action.erase(k);
        }
        auto& args = action["args"];
        if (args.contains("path") && !args.contains("file_path")) {
            args["file_path"] = args["path"];
            args.erase("path");
        }
        if (args.contains("filename") && !args.contains("file_path")) {
            args["file_path"] = args["filename"];
            args.erase("filename");
        }
        if (args.contains("dir") && !args.contains("dir_path")) {
            args["dir_path"] = args["dir"];
            args.erase("dir");
        }
        if (args.contains("directory") && !args.contains("dir_path")) {
            args["dir_path"] = args["directory"];
            args.erase("directory");
        }
        if (args.contains("cmd") && !args.contains("command")) {
            args["command"] = args["cmd"];
            args.erase("cmd");
        }
        if (args.contains("pkg") && !args.contains("package")) {
            args["package"] = args["pkg"];
            args.erase("pkg");
        }
        return action;
    }
    std::string validate_content_syntax(const std::string& file_path,
                                        const std::string& content) {
        fs::path p(file_path);
        std::string ext = p.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (ext != ".cpp" && ext != ".cc" && ext != ".cxx" &&
            ext != ".hpp" && ext != ".h" && ext != ".py") {
            return "";
        }
        std::string key = file_path + "|" + md5_hash(content);
        {
            std::lock_guard<std::mutex> lock(syntax_validation_mutex_);
            auto it = syntax_validation_cache_.find(key);
            if (it != syntax_validation_cache_.end()) {
                return it->second;
            }
        }
        std::string result;
        if (ext == ".py") {
            std::string tmp = "/tmp/agent_syntax_check_" + std::to_string(rand()) + ".py";
            {
                std::ofstream f(tmp);
                f << content;
            }
            std::string out = executor.run_shell_command(
                "python3 -m py_compile \"" + tmp + "\"", 10);
            fs::remove(tmp);
            if (out.rfind("RC:0") != std::string::npos) {
                result = "Syntax validation PASSED.";
            } else {
                result = "Syntax validation FAILED: " + out;
            }
        } else {
            std::string tmp = "/tmp/agent_syntax_check_" + std::to_string(rand()) + ".cpp";
            {
                std::ofstream f(tmp);
                f << content;
            }
            std::string out = executor.run_shell_command(
                "g++ -std=c++20 -fsyntax-only \"" + tmp + "\"", 10);
            fs::remove(tmp);
            if (out.rfind("RC:0") != std::string::npos) {
                result = "Syntax validation PASSED.";
            } else {
                result = "Syntax validation FAILED: " + out;
            }
        }
        {
            std::lock_guard<std::mutex> lock(syntax_validation_mutex_);
            if (syntax_validation_cache_.size() > 64) {
                syntax_validation_cache_.clear();
            }
            syntax_validation_cache_[key] = result;
        }
        return result;
    }
    void parse_and_collect_json(
        const std::string& input,
        std::function<void(const json&)> callback
    ) {
        std::vector<size_t> open_braces;
        std::vector<std::pair<size_t, size_t>> blocks;
        bool in_str = false;
        bool escape = false;
        for (size_t i = 0; i < input.size(); ++i) {
            char ch = input[i];
            if (escape) {
                escape = false;
                continue;
            }
            if (ch == '\\') {
                escape = true;
                continue;
            }
            if (ch == '"') {
                in_str = !in_str;
                continue;
            }
            if (in_str) {
                continue;
            }
            if (ch == '{') {
                open_braces.push_back(i);
            } else if (ch == '}') {
                if (!open_braces.empty()) {
                    blocks.push_back({open_braces.back(), i});
                    open_braces.pop_back();
                }
            }
        }
        std::sort(
            blocks.begin(),
            blocks.end(),
            [](const auto& a, const auto& b) {
                return (a.second - a.first) > (b.second - b.first);
            }
        );
        std::vector<std::pair<size_t, size_t>> consumed;
        std::function<void(const json&)> extract_all =
            [&](const json& j) {
                if (j.is_object()) {
                    callback(j);
                    for (auto it = j.begin(); it != j.end(); ++it) {
                        extract_all(it.value());
                    }
                } else if (j.is_array()) {
                    for (const auto& val : j) {
                        extract_all(val);
                    }
                }
            };
        for (const auto& block_span : blocks) {
            size_t start = block_span.first;
            size_t end = block_span.second;
            bool is_consumed = false;
            for (const auto& c : consumed) {
                if (start >= c.first && end <= c.second) {
                    is_consumed = true;
                    break;
                }
            }
            if (is_consumed) {
                continue;
            }
            std::string text_block =
                input.substr(start, end - start + 1);

            if (text_block.find("tool") == std::string::npos &&
                text_block.find("name") == std::string::npos) {
                continue;
            }
            bool success = false;
            json parsed = json::parse(text_block, nullptr, false);
            if (!parsed.is_discarded()) {
                extract_all(parsed);
                success = true;
            } else {
                std::string repaired = repair_json_fragment(text_block);
                json parsed2 = json::parse(repaired, nullptr, false);
                if (!parsed2.is_discarded()) {
                    extract_all(parsed2);
                    success = true;
                }
            }
            if (success) {
                consumed.push_back(block_span);
            }
        }
    }
    std::string repair_json_fragment(const std::string& fragment) {
        std::string result;
        result.reserve(fragment.size() + fragment.size() / 10);
        bool in_string = false;
        bool escape = false;
        for (size_t i = 0; i < fragment.size(); ++i) {
            char ch = fragment[i];
            if (escape) {
                result += ch;
                escape = false;
                continue;
            }
            if (ch == '\\') {
                result += ch;
                escape = true;
                continue;
            }
            if (ch == '"') {
                in_string = !in_string;
                result += ch;
                continue;
            }
            if (in_string && (ch == '\n' || ch == '\r')) {
                result += (ch == '\n') ? "\\n" : "\\r";
                continue;
            }
            result += ch;
        }
        std::string no_trailing;
        no_trailing.reserve(result.size());
        for (size_t i = 0; i < result.size(); ++i) {
            if (result[i] == ',') {
                bool trailing = false;
                for (size_t j = i + 1; j < result.size(); ++j) {
                    if (std::isspace(
                            static_cast<unsigned char>(result[j]))) {
                        continue;
                    }
                    if (result[j] == '}' || result[j] == ']') {
                        trailing = true;
                    }
                    break;
                }
                if (trailing) {
                    continue;
                }
            }
            no_trailing += result[i];
        }
        std::string fixed;
        fixed.reserve(no_trailing.size());
        for (size_t i = 0; i < no_trailing.size(); ++i) {
            fixed += no_trailing[i];
            if (no_trailing[i] == '{' || no_trailing[i] == ',') {
                size_t j = i + 1;
                while (j < no_trailing.size() &&
                       std::isspace(
                           static_cast<unsigned char>(no_trailing[j]))) {
                    j++;
                }
                if (j < no_trailing.size() &&
                    (std::isalpha(
                         static_cast<unsigned char>(no_trailing[j])) ||
                     no_trailing[j] == '_')) {
                    size_t k = j;
                    while (k < no_trailing.size() &&
                           (std::isalnum(
                                static_cast<unsigned char>(no_trailing[k])) ||
                            no_trailing[k] == '_')) {
                        k++;
                    }
                    size_t l = k;
                    while (l < no_trailing.size() &&
                           std::isspace(
                               static_cast<unsigned char>(no_trailing[l]))) {
                        l++;
                    }
                    if (l < no_trailing.size() &&
                        no_trailing[l] == ':') {
                        fixed += no_trailing.substr(i + 1, j - i - 1);
                        fixed += '"';
                        fixed += no_trailing.substr(j, k - j);
                        fixed += '"';
                        i = k - 1;
                    }
                }
            }
        }
        return fixed;
    }
    json extract_write_file_direct(const std::string& text) {
        size_t fp_pos = text.find("\"file_path\"");
        if (fp_pos == std::string::npos) {
            return json::object();
        }
        size_t colon = text.find(':', fp_pos + 11);
        if (colon == std::string::npos) {
            return json::object();
        }
        size_t quote = text.find('"', colon + 1);
        if (quote == std::string::npos) {
            return json::object();
        }
        std::string file_path;
        bool escape = false;
        size_t end_quote = quote + 1;
        while (end_quote < text.size()) {
            if (escape) {
                file_path += text[end_quote];
                escape = false;
            } else if (text[end_quote] == '\\') {
                escape = true;
            } else if (text[end_quote] == '"') {
                break;
            } else {
                file_path += text[end_quote];
            }
            end_quote++;
        }
        if (end_quote >= text.size()) {
            return json::object();
        }
        size_t content_pos = text.find("\"content\"", end_quote);
        if (content_pos == std::string::npos) {
            return json::object();
        }
        size_t c_colon = text.find(':', content_pos + 9);
        if (c_colon == std::string::npos) {
            return json::object();
        }
        size_t c_quote = text.find('"', c_colon + 1);
        if (c_quote == std::string::npos) {
            return json::object();
        }
        size_t c_end = text.rfind('"');
        if (c_end <= c_quote) {
            return json::object();
        }
        std::string content =
            text.substr(c_quote + 1, c_end - c_quote - 1);
        return {
            {"tool", "write_file"},
            {
                "args",
                {
                    {"file_path", file_path},
                    {"content", content}
                }
            }
        };
    }
    std::vector<json> extract_xml_tool_calls(const std::string& text) {
        std::vector<json> actions;
        size_t pos = 0;
        while ((pos = text.find("<tool_call", pos)) != std::string::npos) {
            size_t tag_end = text.find(">", pos);
            if (tag_end == std::string::npos) {
                break;
            }
            size_t end_tag = text.find("</tool_call>", tag_end);
            if (end_tag == std::string::npos) {
                break;
            }
            std::string full_tag =
                text.substr(pos, tag_end - pos + 1);
            std::string inner =
                text.substr(tag_end + 1, end_tag - tag_end - 1);
            std::string tool;
            size_t name_pos = inner.find("<tool_name>");
            if (name_pos != std::string::npos) {
                size_t name_end =
                    inner.find("</tool_name>", name_pos);
                if (name_end != std::string::npos) {
                    tool = inner.substr(
                        name_pos + 11,
                        name_end - name_pos - 11
                    );
                }
            }
            if (tool.empty()) {
                size_t n_pos = full_tag.find("name=\"");
                if (n_pos != std::string::npos) {
                    size_t n_end =
                        full_tag.find("\"", n_pos + 6);
                    if (n_end != std::string::npos) {
                        tool = full_tag.substr(
                            n_pos + 6,
                            n_end - n_pos - 6
                        );
                    }
                }
            }
            if (tool.empty()) {
                size_t t_pos = full_tag.find("tool=\"");
                if (t_pos != std::string::npos) {
                    size_t t_end =
                        full_tag.find("\"", t_pos + 6);
                    if (t_end != std::string::npos) {
                        tool = full_tag.substr(
                            t_pos + 6,
                            t_end - t_pos - 6
                        );
                    }
                }
            }
            if (!tool.empty()) {
                json args = json::parse(inner, nullptr, false);
                if (args.is_discarded()) {
                    args = json::parse(repair_json_fragment(inner), nullptr, false);
                    if (args.is_discarded()) {
                        args = json::object();
                        size_t path_pos = inner.find("<file_path>");
                        if (path_pos != std::string::npos) {
                            size_t path_end =
                                inner.find("</file_path>", path_pos);
                            if (path_end != std::string::npos) {
                                args["file_path"] = inner.substr(
                                    path_pos + 11,
                                    path_end - path_pos - 11
                                );
                            }
                        }
                        size_t cont_pos = inner.find("<content>");
                        if (cont_pos != std::string::npos) {
                            size_t cont_end =
                                inner.find("</content>", cont_pos);
                            if (cont_end != std::string::npos) {
                                args["content"] = inner.substr(
                                    cont_pos + 9,
                                    cont_end - cont_pos - 9
                                );
                            }
                        }
                        size_t cmd_pos = inner.find("<command>");
                        if (cmd_pos != std::string::npos) {
                            size_t cmd_end =
                                inner.find("</command>", cmd_pos);
                            if (cmd_end != std::string::npos) {
                                args["command"] = inner.substr(
                                    cmd_pos + 9,
                                    cmd_end - cmd_pos - 9
                                );
                            }
                        }
                    }
                }
                json action;
                action["tool"] = tool;
                if (!args.empty()) {
                    action["args"] = args;
                }
                actions.push_back(action);
            }
            pos = end_tag + 12;
        }
        return actions;
    }
};

class TaskDecomposer {
public:
    explicit TaskDecomposer(LLMBackendManager &llm) : llm(llm) {}
    std::vector<std::string> decompose(const std::string &task) {
        json schema = {{"checklist", json::array({"sub-goal 1"})}};
        std::string prompt =
            "Break this task into 3-5 sequential sub-goals. Output ONLY JSON.\nTask: " +
            task;
        json res = llm.structured_chat({{"user", prompt}}, schema, 0.0, 256);
        if (res.contains("checklist") && res["checklist"].is_array()) {
            std::vector<std::string> goals;
            for (auto &g : res["checklist"]) {
                if (g.is_string()) {
                    goals.push_back(g.get<std::string>());
                }
            }
            return goals;
        }
        return {task};
    }
    std::vector<std::string> decompose_plan(const std::vector<json>& plan) {
        if (plan.empty()) {
            return {};
        }
        std::vector<std::string> goals;
        std::set<std::string> seen_files;
        std::set<std::string> seen_dirs;
        bool finish_seen = false;
        auto arg_str = [](const json& action, const char* key) -> std::string {
            if (!action.contains("args") || !action["args"].is_object()) {
                return "";
            }
            const json& args = action["args"];
            if (!args.contains(key) || !args[key].is_string()) {
                return "";
            }
            return args[key].get<std::string>();
        };
        for (const auto& action : plan) {
            if (!action.is_object()) {
                continue;
            }
            std::string tool = action.value("tool", "");
            if (tool == "make_directory") {
                std::string dir = arg_str(action, "dir_path");
                if (!dir.empty() && seen_dirs.insert(dir).second) {
                    goals.push_back("Create directory structure: " + dir);
                }
            } else if (tool == "write_file") {
                std::string fp = arg_str(action, "file_path");
                if (!fp.empty() && seen_files.insert(fp).second) {
                    goals.push_back("Implement " +
                                    fs::path(fp).filename().string());
                }
            } else if (tool == "compile_cpp") {
                std::string fp = arg_str(action, "file_path");
                if (!fp.empty()) {
                    goals.push_back("Compile " +
                                    fs::path(fp).filename().string());
                }
            } else if (tool == "execute_file") {
                std::string fp = arg_str(action, "file_path");
                if (!fp.empty()) {
                    goals.push_back("Execute " +
                                    fs::path(fp).filename().string());
                }
            } else if (tool == "run_shell_command") {
                std::string cmd = arg_str(action, "command");
                if (!cmd.empty()) {
                    std::string short_cmd = cmd.substr(0, 60);
                    std::replace(short_cmd.begin(), short_cmd.end(), '\n', ' ');
                    goals.push_back("Run command: " + short_cmd);
                }
            } else if (tool == "finish") {
                finish_seen = true;
            }
        }
        if (finish_seen) {
            goals.push_back("Validate outputs and finish");
        }
        std::vector<std::string> unique;
        std::set<std::string> seen_goals;
        for (auto &goal : goals) {
            if (seen_goals.insert(goal).second) {
                unique.push_back(goal);
            }
        }
        if (unique.size() > 5) {
            unique.resize(5);
        }
        return unique;
    }
private:
    LLMBackendManager &llm;
};

class DependencyPlanner {
public:
    explicit DependencyPlanner(LLMBackendManager & llm) : llm(llm) {}
    std::vector<json> generate_plan(const std::string & task,
                                    const std::vector<std::string> & expected_outputs,
                                    const std::vector<json> & validation_checks = {}) {
        if (expected_outputs.empty()) return fallback_plan({});
        std::vector<json> plan;
        std::set<std::string> created_dirs;
        for (const auto & fp : expected_outputs) {
            fs::path p(fp);
            std::string parent = p.parent_path().string();
            if (!parent.empty() && parent != "." && created_dirs.insert(parent).second) {
                plan.push_back({{"tool", "make_directory"}, {"args", {{"dir_path", parent}}}});
            }
        }
        std::map<std::string, std::vector<std::string>> file_validation_reqs;
        std::map<std::string, std::string> file_stdin_inputs;
        for (const auto & check : validation_checks) {
            if (check.value("type", "") == "execution") {
                std::string file = check.value("file", "");
                std::string expect = check.value("expect", "");
                std::string stdin_input = check.value("input", "");
                if (!file.empty()) {
                    if (!expect.empty()) file_validation_reqs[file].push_back(expect);
                    if (!stdin_input.empty()) file_stdin_inputs[file] = stdin_input;
                }
            }
        }
        std::string files_desc;
        for (const auto & fp : expected_outputs) {
            std::string ext = fs::path(fp).extension().string();
            std::string lang = "text";
            if (ext == ".py") lang = "Python";
            else if (ext == ".cpp" || ext == ".cc" || ext == ".cxx") lang = "C++";
            files_desc += "- File: " + fp + " (" + lang + ")\n";
            if (file_validation_reqs.count(fp) && !file_validation_reqs[fp].empty()) {
                files_desc += "  CRITICAL VALIDATION: Must print exact substrings to stdout:\n";
                for (const auto & req : file_validation_reqs[fp]) {
                    files_desc += "    - '" + req + "'\n";
                }
            }
            if (file_stdin_inputs.count(fp)) {
                files_desc += "  STDIN INPUT: '" + file_stdin_inputs[fp] + "'\n";
            }
        }
        std::string content_prompt = 
            "Task: " + task + "\n"
            "Generate the complete, functional code for ALL the following files.\n"
            "Files to generate:\n" + files_desc + "\n"
            "Requirements:\n"
            "- Must be fully implemented, no placeholders or TODOs.\n"
            "- Ensure all interfaces between files are perfectly consistent.\n"
            "- Return ONLY the raw code for each file.\n";
        json schema = {
            {"type", "object"},
            {"properties", {
                {"files", {
                    {"type", "array"},
                    {"items", {
                        {"type", "object"},
                        {"properties", {
                            {"file_path", {{"type", "string"}}},
                            {"content", {{"type", "string"}}}
                        }},
                        {"required", json::array({"file_path", "content"})}
                    }}
                }}
            }},
            {"required", json::array({"files"})}
        };
        json parsed = llm.structured_chat({{"user", content_prompt}}, schema, 0.0, 4096);
        bool success = false;
        if (parsed.is_object() && parsed.contains("files") && parsed["files"].is_array()) {
            try {
                for (const auto & file_obj : parsed["files"]) {
                    std::string fp = file_obj.value("file_path", "");
                    std::string content = file_obj.value("content", "");
                    if (content.find("```") != std::string::npos) {
                        size_t start = content.find("```");
                        size_t end = content.rfind("```");
                        if (start != end) {
                            size_t nl = content.find('\n', start);
                            if (nl != std::string::npos && nl < end) {
                                content = content.substr(nl + 1, end - nl - 1);
                            }
                        } else if (start == end) {
                            content.erase(start, 3);
                        }
                    }
                    size_t first = content.find_first_not_of(" \t\n\r");
                    size_t last = content.find_last_not_of(" \t\n\r");
                    if (first != std::string::npos && last != std::string::npos) {
                        content = content.substr(first, last - first + 1);
                    }
                    plan.push_back({{"tool", "write_file"}, {"args", {{"file_path", fp}, {"content", content}}}});
                    std::string ext = fs::path(fp).extension().string();
                    if (ext == ".cpp" || ext == ".cc" || ext == ".cxx") {
                        plan.push_back({{"tool", "compile_cpp"}, {"args", {{"file_path", fp}}}});
                    }
                    plan.push_back({{"tool", "execute_file"}, {"args", {{"file_path", fp}}}});
                }
                success = true;
            } catch (...) {
                success = false;
            }
        }
        if (!success) {
            for (const auto & fp : expected_outputs) {
                std::string ext = fs::path(fp).extension().string();
                std::string lang = "text";
                if (ext == ".py") lang = "Python";
                else if (ext == ".cpp" || ext == ".cc" || ext == ".cxx") lang = "C++";
                std::string content_prompt_ind = "Task: " + task + "\n"
                    "Generate the complete, functional " + lang + " code for the file '" + fp + "'.\n"
                    "Requirements:\n"
                    "- Must be fully implemented, no placeholders or TODOs.\n"
                    "- Return ONLY the raw code. No markdown fences, no explanations.\n";
                if (file_validation_reqs.count(fp) && !file_validation_reqs[fp].empty()) {
                    content_prompt_ind += "\nCRITICAL VALIDATION REQUIREMENTS:\n";
                    content_prompt_ind += "The code will be executed and its stdout will be checked.\n";
                    content_prompt_ind += "You MUST ensure the following exact strings appear in the runtime output:\n";
                    for (const auto & req : file_validation_reqs[fp]) {
                        content_prompt_ind += "- '" + req + "'\n";
                    }
                    content_prompt_ind += "\nIMPORTANT: These strings must be printed to stdout during execution, ";
                    content_prompt_ind += "not just present in comments or variable names.\n";
                    content_prompt_ind += "Add explicit print/cout statements to output these strings.\n";
                }
                if (file_stdin_inputs.count(fp)) {
                    content_prompt_ind += "\nThe code will receive the following stdin input during execution:\n";
                    content_prompt_ind += "'" + file_stdin_inputs[fp] + "'\n";
                    content_prompt_ind += "Ensure the code correctly reads and processes this input.\n";
                }
                ChatResult res_ind = llm.chat({{"user", content_prompt_ind}}, 0.0, 2048);
                std::string content = res_ind.ok ? res_ind.content : "";
                if (content.find("```") != std::string::npos) {
                    size_t start = content.find("```");
                    size_t end = content.rfind("```");
                    if (start != end) {
                        size_t nl = content.find('\n', start);
                        if (nl != std::string::npos && nl < end) {
                            content = content.substr(nl + 1, end - nl - 1);
                        }
                    } else if (start == end) {
                        content.erase(start, 3);
                    }
                }
                size_t first = content.find_first_not_of(" \t\n\r");
                size_t last = content.find_last_not_of(" \t\n\r");
                if (first != std::string::npos && last != std::string::npos) {
                    content = content.substr(first, last - first + 1);
                }
                plan.push_back({{"tool", "write_file"}, {"args", {{"file_path", fp}, {"content", content}}}});
                if (ext == ".cpp" || ext == ".cc" || ext == ".cxx") {
                    plan.push_back({{"tool", "compile_cpp"}, {"args", {{"file_path", fp}}}});
                }
                plan.push_back({{"tool", "execute_file"}, {"args", {{"file_path", fp}}}});
            }
        }
        plan.push_back({{"tool", "finish"}, {"args", {}}});
        return plan;
    }

private:
    LLMBackendManager & llm;
    static std::string join(const std::vector<std::string> & v, const std::string & sep) {
        std::string out;
        for (size_t i = 0; i < v.size(); ++i) {
            if (i) out += sep;
            out += v[i];
        }
        return out;
    }
    static void collect_actions(const json & node, std::vector<json> & out) {
        if (node.is_array()) {
            for (const auto & child : node)
                collect_actions(child, out);
        } else if (node.is_object()) {
            if (node.contains("tool") && node["tool"].is_string()) {
                out.push_back(node);
            } else {
                for (auto it = node.begin(); it != node.end(); ++it)
                    collect_actions(it.value(), out);
            }
        }
    }
    static std::vector<json> extract_plan_actions(const json & plan) {
        std::vector<json> actions;
        if (plan.is_object()) {
            if (plan.contains("actions"))
                collect_actions(plan["actions"], actions);
            else if (plan.contains("plan"))
                collect_actions(plan["plan"], actions);
            else
                collect_actions(plan, actions);
        } else if (plan.is_array()) {
            collect_actions(plan, actions);
        }
        return actions;
    }
    static json normalize_action(const json & a) {
        if (!a.is_object()) return json();
        json action = a;
        if (!action.contains("tool") || !action["tool"].is_string()) {
            if (action.contains("name") && action["name"].is_string()) {
                action["tool"] = action["name"];
            } else {
                return json();
            }
        }
        if (!action.contains("args") || !action["args"].is_object())
            action["args"] = json::object();
        for (auto & [k, v] : action.items()) {
            if (k != "tool" && k != "args")
                action["args"][k] = v;
        }
        return action;
    }
    std::vector<json> fallback_plan(const std::vector<std::string> & files) {
        std::vector<json> actions;
        std::set<std::string> seen_dirs;
        for (auto & fp : files) {
            fs::path p(fp);
            std::string parent = p.parent_path().string();
            if (!parent.empty() && parent != "." && !seen_dirs.count(parent)) {
                actions.push_back({{"tool", "make_directory"}, {"args", {{"dir_path", parent}}}});
                seen_dirs.insert(parent);
            }
        }
        return actions;
    }
    std::vector<json> fix_files(std::vector<json> actions, const std::vector<std::string> & expected) {
        std::set<std::string> names;
        for (auto & p : expected) names.insert(fs::path(p).filename().string());
        for (auto & act : actions) {
            if (act.value("tool", "") == "write_file") {
                std::string fp = act["args"].value("file_path", "");
                if (!fp.empty() && names.count(fs::path(fp).filename().string())) {
                    for (auto & real : expected) {
                        if (fs::path(real).filename() == fs::path(fp).filename()) {
                            act["args"]["file_path"] = real;
                            break;
                        }
                    }
                }
            }
        }
        return actions;
    }
};

class TestSuite {
public:
    TestSuite() : cases(default_cases()) {}
    const std::vector<TestCase>& get_cases() const {
        return cases;
    }

private:
    std::vector<TestCase> cases;
    static std::vector<TestCase> default_cases() {
        std::vector<TestCase> cases;
        cases.push_back({
            "T1", "Automated Debugging via Log Analysis",
            "Automated Debugging via Log Analysis",
            "The file memory_leak.cpp already exists and contains a segfault bug. "
            "1. Execute the file to observe the crash. 2. Read execution_traceback.log. "
            "3. Fix the bug so the file runs and exits with code 0. Do NOT simply replace it with unrelated code.",
            {"memory_leak.cpp"},
            {{{"type", "execution"}, {"file", "memory_leak.cpp"}}},
            6, 120
        });
        cases.push_back({
            "T2", "Concurrent Hashing",
            "Parallel processing with concurrent.futures",
            "Create concurrent_hash.py scanning for .txt/.py files, calculate SHA-256 in parallel. "
            "Store {filename: hash} dict. Print execution time and verify count matches.",
            {"concurrent_hash.py"},
            {{{"type", "execution"}, {"file", "concurrent_hash.py"}, {"expect", "SHA-256"}}},
            4, 120
        });
        cases.push_back({
            "T3", "Multi-Language Data Bridge",
            "Multi-Language Data Bridge",
            "Create a C++ program named generator.cpp that simulates 10^6 particle interactions and saves "
            "the state to a data.json file using the nlohmann/json library. Subsequently, create a Python "
            "script analyzer.py that reads this JSON file, calculates the mean energy of the particles, and "
            "generates a summary report. This tests the agent ability to manage dependencies across different "
            "languages and ensure data consistency between processes.",
            {"generator.cpp", "analyzer.py"},
            {{{"type", "execution"}, {"file", "analyzer.py"}, {"expect", "Mean"}}},
            6, 120
        });
        cases.push_back({
            "T4", "Concurrent Data Processing with Thread Safety",
            "Concurrent Data Processing with Thread Safety",
            "Create a CSV file named \"data.csv\" with 10,000 rows of random integers. Implement a C++ "
            "program \"data_processor.cpp\" that reads this file, processes it in parallel using threads, "
            "and outputs the processed results to \"output.json\". Ensure thread safety with appropriate "
            "synchronization mechanisms.",
            {"data_processor.cpp", "output.json"},
            {{{"type", "execution"}, {"file", "data_processor.cpp"}, {"expect", "Processed"}}},
            8, 120
        });
        cases.push_back({
            "T5", "Secure Input Validation",
            "Secure Input Validation",
            "Create a single Python script named secure_query.py that demonstrates secure SQLite querying. "
            "The script MUST: (1) Create an SQLite database file and a 'users' table with at least one row "
            "(e.g., name='John Doe'), (2) Read lines from stdin in a loop until 'exit' is received, "
            "(3) Use parameterized queries to search for users by name, (4) Print 'Query successful' "
            "when a query executes without error, (5) Attempt a common SQL injection attack on itself "
            "and show the attack is blocked. The script will be executed with stdin 'John Doe\\nexit\\n' "
            "and the output MUST contain 'Query successful'.",
            {"secure_query.py"},
            {{{"type", "execution"}, {"file", "secure_query.py"},
              {"input", "John Doe\nexit\n"}, {"expect", "Query successful"}}},
            5, 120
        });
        cases.push_back({
            "T6", "Monte Carlo Integration",
            "Probabilistic numerical methods",
            "Create a c++ program monte_carlo.cpp to estimate ∫sin(x)dx from 0 to π using 1,000,000 random points. "
            "Compare estimate to analytical value 2. Print absolute error.",
            {"monte_carlo.cpp"},
            {{{"type", "execution"}, {"file", "monte_carlo.cpp"}, {"expect", "Absolute error"}}},
            5, 120
        });
        cases.push_back({
            "T7", "Import Restructuring",
            "Handle directory moves and import updates",
            "1. Create math_lib/operations.py with power(a,b) function. Create app.py importing it and "
            "printing 2**10. 2. Move operations.py to core/utils/, update imports, verify output still 1024.",
            {"app.py", "core/utils/operations.py"},
            {{{"type", "execution"}, {"file", "app.py"}, {"expect", "1024"}}},
            8, 120
        });
        cases.push_back({
            "T8", "Vending Machine OOP",
            "Custom exceptions and state management",
            "Create vending.py with VendingMachine class and InsufficientFundsError exception. Demonstrate: "
            "deposit $2.00, try buying a $2.50 item, catch the InsufficientFundsError, and print the EXACT text "
            "'InsufficientFundsError' (this exact string is required for validation). Then deposit $1 more and "
            "buy successfully. Finally print the remaining balance and inventory.",
            {"vending.py"},
            {{{"type", "execution"}, {"file", "vending.py"}, {"expect", "InsufficientFundsError"}}},
            6, 120
        });
        cases.push_back({
            "T9", "Pi Approximation",
            "Numerical methods and convergence",
            "Create pi_approx.py using Leibniz formula: π = 4 * Σ((-1)^n/(2n+1)). Iterate until "
            "|approximation - math.pi| < 10^-5. Print iterations needed.",
            {"pi_approx.py"},
            {{{"type", "execution"}, {"file", "pi_approx.py"}, {"expect", "iterations"}}},
            4, 120
        });
        cases.push_back({
            "T10", "Grid BFS Pathfinding",
            "Graph traversal and obstacle handling",
            "Create grid_bfs.py with a 10x10 grid containing some obstacles. Implement BFS from (0,0) to (9,9). "
            "Print the path using the EXACT text 'coordinates' (for example: 'Path coordinates: [...]') and the step count. "
            "If no path exists, print 'No path found.'.",
            {"grid_bfs.py"},
            {{{"type", "execution"}, {"file", "grid_bfs.py"}, {"expect", "coordinates"}}},
            4, 120
        });
        cases.push_back({
            "T11", "SQLite Relational Query",
            "Complex JOINs and subqueries",
            "Create SQLite database with Departments and Employees tables. Insert 3 departments, 10 employees. Query: \n"
            "find employees earning more than their dept average. Save to high_earners.json. Use proper foreign keys.\n",
            {"high_earners.json"},
            {{{"type", "file_exists"}, {"path", "high_earners.json"}}},
            8, 120
        });
        cases.push_back({
            "T12", "Memoization Decorator",
            "Higher-order functions and performance",
            "Create fibonacci_memoized.py with memoize decorator and apply to recursive Fibonacci. "
            "Calculate F(50) with timing. Compare to F(30) without decorator.",
            {"fibonacci_memoized.py"},
            {{{"type", "execution"}, {"file", "fibonacci_memoized.py"}, {"expect", "F(50)"}}},
            6, 120
        });
        cases.push_back({
            "T13", "Cross-File Refactoring",
            "Cross-File Refactoring",
            "Task the agent with renaming a specific class or utility function in a multi-file project. "
            "For example, it must create a directory src/math, move several mathematical functions from a "
            "single utils.py into separate modules, and then update the import statements in an app.py "
            "located in the root directory. This evaluates the efficiency of the FileManager in handling "
            "path updates and recursive directory searches.",
            {"app.py", "utils.py"},
            {{{"type", "execution"}, {"file", "app.py"}, {"expect", "power"}}},
            4, 120
        });
        cases.push_back({
            "T14", "Data Cleaning Pipeline",
            "Handle dirty CSV data with type validation",
            "Create raw_data.csv (Name, Age, Salary) with 6 rows including non-numeric ages/empty salaries. "
            "Create cleaner.py to filter invalid rows, calculate average salary, save to processed_data.json.",
            {"raw_data.csv", "cleaner.py", "processed_data.json"},
            {{{"type", "file_exists"}, {"path", "processed_data.json"}}},
            6, 120
        });
        cases.push_back({
            "T15", "Networked Resource Fetching",
            "Networked Resource Fetching",
            "Require the agent to use the curl capability to interface with a public API to retrieve "
            "current weather or financial data. The agent must then use openssl/md5 to create a unique "
            "checksum of the response to verify data integrity before processing. This evaluates the "
            "ability to handle external libraries and asynchronous-like behavior within the execution environment.",
            {"api_fetch.cpp"},
            {{{"type", "execution"}, {"file", "api_fetch.cpp"}, {"expect", "Checksum"}}},
            4, 120
        });
        cases.push_back({
            "T16", "LRU Cache with TTL",
            "Data structures and time-bound expiration",
            "Implement an LRU (Least Recently Used) cache with TTL (Time-To-Live) in lru_cache.py. "
            "Requirements: O(1) get and put. Entries expire automatically after TTL seconds. "
            "Must support: get(key), put(key, value, ttl), size(). Print demonstration showing "
            "eviction due to capacity and expiration due to TTL.",
            {"lru_cache.py"},
            {
                {{"type", "execution"}, {"file", "lru_cache.py"}, {"expect", "LRU"}},
                {{"type", "execution"}, {"file", "lru_cache.py"}, {"expect", "expired"}},
                {{"type", "execution"}, {"file", "lru_cache.py"}, {"expect", "evicted"}},
                {{"type", "execution"}, {"file", "lru_cache.py"}, {"expect", "O(1)"}}
            },
            6, 120
        });
        cases.push_back({
            "T17", "Topological Sort with Cycle Detection",
            "Graph algorithms and error handling",
            "Create topo_sort.py that reads a DAG from edges list and outputs a valid topological ordering. "
            "If a cycle is detected, raise CycleError with message 'Graph contains a cycle'. "
            "Use Kahn's algorithm (BFS-based). Test with: A->B, A->C, B->D, C->D.",
            {"topo_sort.py"},
            {
                {{"type", "execution"}, {"file", "topo_sort.py"}, {"expect", "Topological"}},
                {{"type", "execution"}, {"file", "topo_sort.py"}, {"expect", "CycleError"}}
            },
            6, 120
        });
        cases.push_back({
            "T18", "Mini Regex Engine",
            "Finite automata and string algorithms",
            "Create regex_engine.py implementing a basic regex matcher supporting: . (any char), * (zero+ of prev), "
            "| (alternation), and literal characters. Do NOT use re module. Implement match(pattern, text) returning bool. "
            "Demonstrate: match('a*b', 'aaab')==True, match('a.b', 'acb')==True, match('a|b', 'b')==True.",
            {"regex_engine.py"},
            {
                {{"type", "execution"}, {"file", "regex_engine.py"}, {"expect", "True"}},
                {{"type", "execution"}, {"file", "regex_engine.py"}, {"expect", "False"}}
            },
            6, 120
        });
        cases.push_back({
            "T19", "Producer-Consumer with Backpressure",
            "Advanced concurrency and queue management",
            "Create producer_consumer.py with bounded queue (maxsize=5). Producer generates 20 items. "
            "Consumer processes items. If queue full, producer blocks. If queue empty, consumer blocks. "
            "Use threading.Condition for synchronization. Print 'produced X' and 'consumed X' for each item. "
            "Final output must contain 'Done: 20 produced, 20 consumed'.",
            {"producer_consumer.py"},
            {
                {{"type", "execution"}, {"file", "producer_consumer.py"},
                 {"expect", "Done: 20 produced, 20 consumed"}},
                {{"type", "execution"}, {"file", "producer_consumer.py"}, {"expect", "produced"}},
                {{"type", "execution"}, {"file", "producer_consumer.py"}, {"expect", "consumed"}}
            },
            6, 120
        });
        cases.push_back({
            "T20", "JSON Parser from Scratch",
            "Recursive descent parsing and tokenization",
            "Create json_parser.py that parses a JSON string WITHOUT using json module. Support: objects, arrays, "
            "strings, numbers, booleans, null. Return native Python dict/list/str/int/float/None. "
            "Demonstrate parsing '{\"name\": \"Alice\", \"age\": 30, \"active\": true, \"nested\": {\"key\": null}}'. "
            "Print parsed['name'] == 'Alice'.",
            {"json_parser.py"},
            {
                {{"type", "execution"}, {"file", "json_parser.py"}, {"expect", "Alice"}},
                {{"type", "execution"}, {"file", "json_parser.py"}, {"expect", "null"}},
                {{"type", "execution"}, {"file", "json_parser.py"}, {"expect", "nested"}}
            },
            6, 120
        });
        cases.push_back({
            "T21", "Streaming Median Calculator",
            "Heaps and online algorithms",
            "Create streaming_median.py that maintains median of a stream using two heaps (max-heap for lower half, "
            "min-heap for upper half). Implement add(num) and get_median(). Demonstrate with stream: [5, 15, 1, 3, 8, 7, 9, 10]. "
            "After each insertion, print current median. Final output must show medians: 5.0, 10.0, 5.0, 4.0, 5.0, 6.0, 7.0, 7.5.",
            {"streaming_median.py"},
            {
                {{"type", "execution"}, {"file", "streaming_median.py"}, {"expect", "7.5"}},
                {{"type", "execution"}, {"file", "streaming_median.py"}, {"expect", "5.0"}},
                {{"type", "execution"}, {"file", "streaming_median.py"}, {"expect", "10.0"}},
                {{"type", "execution"}, {"file", "streaming_median.py"}, {"expect", "O(log n)"}}
            },
            6, 120
        });
        cases.push_back({
            "T22", "Template Engine",
            "String interpolation and control flow",
            "Create template_engine.py with a simple template renderer. Support: {{var}} variable substitution, "
            "{% for item in items %}...{% endfor %} loops, and {% if condition %}...{% endif %} conditionals. "
            "No external template libraries. Render: 'Hello {{name}}! You have {{count}} messages.' "
            "with name='Alice', count=5. Output must contain 'Hello Alice! You have 5 messages.'",
            {"template_engine.py"},
            {
                {{"type", "execution"}, {"file", "template_engine.py"},
                 {"expect", "Hello Alice! You have 5 messages."}},
                {{"type", "execution"}, {"file", "template_engine.py"}, {"expect", "for"}},
                {{"type", "execution"}, {"file", "template_engine.py"}, {"expect", "if"}}
            },
            6, 120
        });
        cases.push_back({
            "T23", "Token Bucket Rate Limiter",
            "Algorithm design and time-based state",
            "Create rate_limiter.py implementing a token bucket algorithm. Class TokenBucket with capacity and refill_rate. "
            "allow_request(tokens_needed) returns True if tokens available, else False. Tokens refill over time. "
            "Demonstrate: bucket=capacity=10, rate=1/sec. Request 5 (True), request 7 (False), wait 2s, request 7 (True). "
            "Output must show 'Allowed: True/False' pattern matching expected behavior.",
            {"rate_limiter.py"},
            {
                {{"type", "execution"}, {"file", "rate_limiter.py"}, {"expect", "Allowed"}},
                {{"type", "execution"}, {"file", "rate_limiter.py"}, {"expect", "False"}}
            },
            6, 120
        });
        cases.push_back({
            "T24", "Sudoku Solver with Constraint Propagation",
            "Backtracking and constraint satisfaction",
            "Create sudoku_solver.py that solves 9x9 Sudoku using backtracking + constraint propagation (forward checking). "
            "Input is a grid with 0 for empty cells. Print solved grid. Must solve hard puzzle in <2 seconds. "
            "Puzzle: [[5,3,0,0,7,0,0,0,0],[6,0,0,1,9,5,0,0,0],[0,9,8,0,0,0,0,6,0],[8,0,0,0,6,0,0,0,3],[4,0,0,8,0,3,0,0,1],[7,0,0,0,2,0,0,0,6],[0,6,0,0,0,0,2,8,0],[0,0,0,4,1,9,0,0,5],[0,0,0,0,8,0,0,7,9]]",
            {"sudoku_solver.py"},
            {
                {{"type", "execution"}, {"file", "sudoku_solver.py"}, {"expect", "Solved"}},
                {{"type", "execution"}, {"file", "sudoku_solver.py"}, {"expect", "8"}},
                {{"type", "execution"}, {"file", "sudoku_solver.py"}, {"expect", "2"}}
            },
            6, 120
        });
        cases.push_back({
            "T25", "Custom Event Loop",
            "Async primitives and callback scheduling",
            "Create event_loop.py implementing a minimal async event loop from scratch. No asyncio module. "
            "Support: call_soon(callback), run_until_complete(), and basic sleep(delay) coroutine. "
            "Demonstrate scheduling 3 tasks that print 'Task N done' after staggered delays. "
            "Output must contain 'Task 1 done', 'Task 2 done', 'Task 3 done' in correct order.",
            {"event_loop.py"},
            {
                {{"type", "execution"}, {"file", "event_loop.py"}, {"expect", "Task 3 done"}},
                {{"type", "execution"}, {"file", "event_loop.py"}, {"expect", "Task 1 done"}},
                {{"type", "execution"}, {"file", "event_loop.py"}, {"expect", "Task 2 done"}}
            },
            6, 120
        });
        cases.push_back({
            "T26", "Multi-Stage Buggy Pipeline Diagnosis",
            "Long-horizon debugging across multiple interdependent files",
            "You must build and then repair a three-stage data pipeline that is intentionally fragile.\n"
            "1. Create generator.py that writes 100 random (id, value) pairs to stage1.csv.\n"
            "2. Create transformer.py that reads stage1.csv, doubles every value, and writes stage2.json.\n"
            "3. Create aggregator.py that reads stage2.json, computes the sum of all values, and prints "
            "'TOTAL: <sum>' (exact prefix required).\n"
            "4. After the pipeline is written, deliberately introduce (or leave) two realistic bugs: "
            "(a) transformer.py uses the wrong column name or off-by-one indexing, "
            "(b) aggregator.py forgets to convert strings to numbers or mishandles the JSON structure.\n"
            "5. Execute the full pipeline, observe the failures, diagnose root causes by reading the "
            "actual output/error messages, and apply the SMALLEST possible patches. Do not rewrite the "
            "files from scratch once they exist.\n"
            "Final requirement: running aggregator.py (after the upstream stages have produced their files) "
            "must print a line starting with 'TOTAL: ' followed by a numeric value.",
            {"generator.py", "transformer.py", "aggregator.py", "stage1.csv", "stage2.json"},
            {
                {{"type", "execution"}, {"file", "aggregator.py"}, {"expect", "TOTAL:"}},
                {{"type", "file_exists"}, {"path", "stage1.csv"}},
                {{"type", "file_exists"}, {"path", "stage2.json"}}
            },
            12, 180
        });
        cases.push_back({
            "T27", "Hidden-Constraint Priority Queue",
            "Spec → implementation with latent algorithmic constraints",
            "Implement a priority queue in priority_queue.py that supports insert(x) and extract_min().\n"
            "The public interface must be a class named PriorityQueue.\n"
            "Hidden but mandatory constraints (discover them by failing validation if violated):\n"
            "- extract_min must be O(log n); a linear scan will time out on the validation input.\n"
            "- Duplicate values must be allowed and returned correctly.\n"
            "- After 10 000 random inserts of integers in [0, 1e6], followed by 5 000 extract_min calls, "
            "the sequence of extracted values must be non-decreasing.\n"
            "- The program must print the exact line 'PriorityQueue OK' after a successful self-test "
            "that exercises the above properties.\n"
            "Do not use the heapq module or any other library heap; implement the heap yourself "
            "(array-based binary heap is expected).",
            {"priority_queue.py"},
            {
                {{"type", "execution"}, {"file", "priority_queue.py"}, {"expect", "PriorityQueue OK"}},
                {{"type", "execution"}, {"file", "priority_queue.py"}, {"expect", "OK"}}
            },
            10, 150
        });
        cases.push_back({
            "T28", "Partial Solution Recovery — Graph Distance",
            "Diagnose and finish a 70%-correct but buggy implementation",
            "Create a file graph_distance.py that is intentionally incomplete / buggy, then repair it.\n"
            "Requirements for the final working program:\n"
            "- Represent an undirected unweighted graph as an adjacency list.\n"
            "- Implement bfs_distance(start, goal) returning the shortest-path length "
            "(or -1 if unreachable).\n"
            "- The self-test must build the graph with edges: 0-1, 0-2, 1-3, 2-3, 3-4, "
            "then print the exact lines:\n"
            "  dist(0,4)=2\n"
            "  dist(1,4)=2\n"
            "  dist(4,0)=2\n"
            "  dist(0,5)=-1\n"
            "First write a version that contains at least two of the following realistic bugs "
            "(off-by-one in distance, missing visited set, treating the graph as directed, "
            "returning the path instead of the length, etc.). Execute it, observe the wrong output, "
            "then apply minimal fixes until the four printed lines match exactly. "
            "Do not discard the existing code and rewrite from scratch after the first version exists.",
            {"graph_distance.py"},
            {
                {{"type", "execution"}, {"file", "graph_distance.py"}, {"expect", "dist(0,4)=2"}},
                {{"type", "execution"}, {"file", "graph_distance.py"}, {"expect", "dist(0,5)=-1"}},
                {{"type", "execution"}, {"file", "graph_distance.py"}, {"expect", "dist(1,4)=2"}}
            },
            10, 150
        });
        cases.push_back({
            "T29", "Four-Stage Report Pipeline",
            "Long dependency chain with consistency requirements",
            "Build a four-stage reporting pipeline. Each stage consumes the previous stage's artifact.\n"
            "Stage A — create seed_data.csv containing 20 rows of (name, score) with scores in 0..100.\n"
            "Stage B — create normalize.py that reads seed_data.csv, normalizes scores to [0,1], "
            "and writes normalized.json (list of objects with keys 'name' and 'norm_score').\n"
            "Stage C — create ranker.py that reads normalized.json, sorts by descending norm_score, "
            "and writes ranking.txt (one 'rank. name: score' line per entry).\n"
            "Stage D — create summary.py that reads ranking.txt and prints exactly two lines:\n"
            "  TOP: <name of first entry>\n"
            "  COUNT: 20\n"
            "All four artifacts (seed_data.csv, normalized.json, ranking.txt, and the final stdout) "
            "must be consistent with one another. If any intermediate file is missing or malformed, "
            "later stages will fail; diagnose and repair the broken stage rather than regenerating "
            "everything. Final validation runs summary.py and checks for the TOP: and COUNT: lines.",
            {"seed_data.csv", "normalize.py", "normalized.json", "ranker.py", "ranking.txt", "summary.py"},
            {
                {{"type", "execution"}, {"file", "summary.py"}, {"expect", "TOP:"}},
                {{"type", "execution"}, {"file", "summary.py"}, {"expect", "COUNT: 20"}},
                {{"type", "file_exists"}, {"path", "normalized.json"}},
                {{"type", "file_exists"}, {"path", "ranking.txt"}}
            },
            14, 200
        });
        cases.push_back({
            "T30", "Noisy Log Diagnosis — Root Cause Isolation",
            "Multi-turn diagnosis under distracting error messages",
            "Create a small C++ program calc.cpp that is supposed to read two integers from stdin, "
            "compute (a + b) * (a - b), and print the result followed by the exact token 'RESULT'.\n"
            "Also create a companion script run_calc.py that:\n"
            "  - writes a temporary input file,\n"
            "  - compiles calc.cpp,\n"
            "  - runs the binary,\n"
            "  - captures stdout/stderr into a log file named calc_run.log.\n"
            "Intentionally introduce a single root-cause bug in calc.cpp (e.g., integer overflow on "
            "large inputs, off-by-one in the formula, wrong operator, or failure to flush output). "
            "In addition, make run_calc.py emit at least two distracting / secondary messages "
            "(warnings about temporary files, non-fatal compiler notes, or benign stderr lines) "
            "so that the log contains multiple error-like signatures.\n"
            "The agent must:\n"
            "1. Execute the pipeline and obtain a failing log,\n"
            "2. Read calc_run.log,\n"
            "3. Distinguish the real root cause from the noise,\n"
            "4. Apply a minimal fix to calc.cpp only,\n"
            "5. Re-run until the output contains both a correct numeric result and the token 'RESULT'.\n"
            "Validation executes run_calc.py (or the final binary) and requires the substring 'RESULT'.",
            {"calc.cpp", "run_calc.py", "calc_run.log"},
            {
                {{"type", "execution"}, {"file", "run_calc.py"}, {"expect", "RESULT"}},
                {{"type", "file_exists"}, {"path", "calc.cpp"}}
            },
            12, 180
        });
        return cases;
    }
};

class CodingAgentUtils {
public:
    static std::string validation_file_signature(const std::string &fp) {
        std::error_code ec;
        auto ftime = fs::last_write_time(fp, ec);
        if (ec) return "missing";
        auto size = fs::file_size(fp, ec);
        if (ec) return "missing";
        return std::to_string(ftime.time_since_epoch().count()) + ":" + std::to_string(size);
    }
    static std::string sanitize_json_string_literals(const std::string &text) {
        std::string out;
        out.reserve(text.size() + text.size() / 4);
        bool in_string = false;
        bool escaped = false;
        for (size_t i = 0; i < text.size(); ++i) {
            const char c = text[i];
            if (in_string) {
                if (escaped) { out += c; escaped = false; } 
                else if (c == '\\') { out += c; escaped = true; } 
                else if (c == '"') {
                    bool is_end = false;
                    size_t j = i + 1;
                    while (j < text.size() && (text[j] == ' ' || text[j] == '\n' || text[j] == '\r' || text[j] == '\t')) ++j;
                    if (j == text.size() || text[j] == '}' || text[j] == ']' || text[j] == ',' || text[j] == ':') is_end = true;
                    if (is_end) { out += c; in_string = false; } 
                    else { out += "\\\""; }
                } 
                else if (c == '\n') out += "\\n"; 
                else if (c == '\r') out += "\\r"; 
                else if (c == '\t') out += "\\t"; 
                else out += c;
            } else {
                if (c == '"') { out += c; in_string = true; } 
                else out += c;
            }
        }
        return out;
    }
    static std::string repair_json(const std::string &input) {
        std::string out;
        out.reserve(input.size() * 1.2);
        bool in_string = false;
        bool escape = false;
        bool in_key = true;
        int brace_depth = 0;
        int bracket_depth = 0;
        for (size_t i = 0; i < input.size(); ++i) {
            char c = input[i];
            if (escape) { out += c; escape = false; continue; }
            if (in_string) {
                if (c == '\\') { out += c; escape = true; } 
                else if (c == '"' || c == '\'') { out += '"'; in_string = false; } 
                else if (c == '\n') out += "\\n"; 
                else if (c == '\r') out += "\\r"; 
                else if (c == '\t') out += "\\t"; 
                else out += c;
            } else {
                if (c == '"') { out += c; in_string = true; } 
                else if (c == '\'') { out += '"'; in_string = true; } 
                else if (c == '{') { out += c; brace_depth++; in_key = true; } 
                else if (c == '}') {
                    while (!out.empty() && std::isspace((unsigned char)out.back())) out.pop_back();
                    if (!out.empty() && out.back() == ',') out.pop_back();
                    out += c; brace_depth--; in_key = false;
                } 
                else if (c == '[') { out += c; bracket_depth++; } 
                else if (c == ']') {
                    while (!out.empty() && std::isspace((unsigned char)out.back())) out.pop_back();
                    if (!out.empty() && out.back() == ',') out.pop_back();
                    out += c; bracket_depth--;
                } 
                else if (c == ',') { out += c; in_key = (brace_depth > 0 && bracket_depth == 0); } 
                else if (c == ':') { out += c; in_key = false; } 
                else if (in_key && std::isalpha((unsigned char)c)) {
                    out += '"';
                    while (i < input.size() && (std::isalnum((unsigned char)input[i]) || input[i] == '_')) out += input[i++];
                    out += '"'; i--;
                } 
                else if (c == '/' && i + 1 < input.size() && input[i + 1] == '/') {
                    while (i < input.size() && input[i] != '\n') i++;
                } 
                else out += c;
            }
        }
        while (bracket_depth > 0) { out += ']'; bracket_depth--; }
        while (brace_depth > 0) { out += '}'; brace_depth--; }
        return out;
    }
    static void extract_tools_from_json(const json &j, std::vector<json> &actions) {
        if (j.is_object()) {
            if (j.contains("failed_generation") && j["failed_generation"].is_string()) {
                try {
                    json fg = json::parse(j["failed_generation"].get<std::string>());
                    extract_tools_from_json(fg, actions);
                } catch (...) {
                    try {
                        std::string sanitized = sanitize_json_string_literals(j["failed_generation"].get<std::string>());
                        json fg = json::parse(sanitized);
                        extract_tools_from_json(fg, actions);
                    } catch (...) {}
                }
            }
            if (j.contains("error") && j["error"].is_object()) extract_tools_from_json(j["error"], actions);
            if (j.contains("tool") && (j["tool"].is_string() || j["tool"].is_object())) { actions.push_back(j); return; }
            if (j.contains("action") && j["action"].is_object() && j["action"].contains("tool")) { actions.push_back(j["action"]); return; }
            if (j.contains("tool_call") && j["tool_call"].is_object() && j["tool_call"].contains("tool")) { actions.push_back(j["tool_call"]); return; }
            if (j.contains("function") && j["function"].is_object() && j["function"].contains("name")) {
                json act;
                act["tool"] = j["function"]["name"].get<std::string>();
                json args = j["function"].value("arguments", json::object());
                if (args.is_string()) {
                    try { args = json::parse(args.get<std::string>()); } 
                    catch (...) {
                        try { args = json::parse(sanitize_json_string_literals(args.get<std::string>())); } 
                        catch (...) { args = json::object(); }
                    }
                }
                if (args.is_object() && args.contains("tool") && args.contains("args")) actions.push_back(args);
                else { act["args"] = args; actions.push_back(act); }
                return;
            }
            if (j.contains("name") && j["name"].is_string() && j.contains("arguments")) {
                json act;
                act["tool"] = j["name"].get<std::string>();
                json args = j["arguments"];
                if (args.is_string()) {
                    try { args = json::parse(args.get<std::string>()); } 
                    catch (...) {
                        try { args = json::parse(sanitize_json_string_literals(args.get<std::string>())); } 
                        catch (...) { args = json::object(); }
                    }
                }
                if (args.is_object() && args.contains("tool") && args.contains("args")) actions.push_back(args);
                else { act["args"] = args; actions.push_back(act); }
                return;
            }
            for (auto it = j.begin(); it != j.end(); ++it) extract_tools_from_json(it.value(), actions);
        } else if (j.is_array()) {
            for (const auto &item : j) extract_tools_from_json(item, actions);
        }
    }
    static std::vector<json> extract_actions_robust(const std::string &text) {
        std::vector<json> actions;
        auto add_action_unique = [&](const json &act) {
            for (const auto &existing : actions) if (existing == act) return;
            actions.push_back(act);
        };
        auto try_parse_and_extract = [&](const std::string &candidate) {
            std::vector<json> extracted;
            json root = json::parse(candidate, nullptr, false);
            if (!root.is_discarded()) {
                extract_tools_from_json(root, extracted);
            } else {
                std::string sanitized = sanitize_json_string_literals(candidate);
                json s_root = json::parse(sanitized, nullptr, false);
                if (!s_root.is_discarded()) {
                    extract_tools_from_json(s_root, extracted);
                } else {
                    std::string repaired = repair_json(candidate);
                    json r_root = json::parse(repaired, nullptr, false);
                    if (!r_root.is_discarded()) extract_tools_from_json(r_root, extracted);
                }
            }
            for (const auto &act : extracted) add_action_unique(act);
        };
        size_t pos = 0;
        const size_t len = text.size();
        while (pos + 2 < len) {
            size_t start = text.find("```", pos);
            if (start == std::string::npos) break;
            size_t lang_end = text.find('\n', start + 3);
            size_t content_start = (lang_end == std::string::npos) ? start + 3 : lang_end + 1;
            size_t end = text.find("```", content_start);
            if (end == std::string::npos) break;
            std::string block = text.substr(content_start, end - content_start);
            size_t first = block.find_first_not_of(" \t\n\r");
            size_t last = block.find_last_not_of(" \t\n\r");
            if (first != std::string::npos && last != std::string::npos) {
                block = block.substr(first, last - first + 1);
            }
            try_parse_and_extract(block);
            pos = end + 3;
        }
        if (!actions.empty()) return actions;
        try_parse_and_extract(text);
        if (!actions.empty()) return actions;
        std::string repaired_text = repair_json(text);
        size_t start = 0;
        while (start < repaired_text.size()) {
            size_t obj_start = repaired_text.find_first_of("{[", start);
            if (obj_start == std::string::npos) break;
            char open_char = repaired_text[obj_start];
            char close_char = (open_char == '{') ? '}' : ']';
            int depth = 0;
            bool in_string = false;
            bool escape = false;
            size_t i = obj_start;
            for (; i < repaired_text.size(); ++i) {
                char c = repaired_text[i];
                if (escape) { escape = false; continue; }
                if (c == '\\') { escape = true; continue; }
                if (c == '"') { in_string = !in_string; continue; }
                if (!in_string) {
                    if (c == open_char) depth++;
                    else if (c == close_char) {
                        depth--;
                        if (depth == 0) {
                            std::string block = repaired_text.substr(obj_start, i - obj_start + 1);
                            try_parse_and_extract(block);
                            start = i + 1;
                            break;
                        }
                    }
                }
            }
            if (i == repaired_text.size()) break;
        }
        return actions;
    }

private:
    static std::string repair_string_newlines_impl(const std::string &src, bool support_triple) {
        std::string out;
        out.reserve(src.size() * 2);
        size_t i = 0;
        int in_str = 0;
        bool escaped = false;
        while (i < src.size()) {
            const char c = src[i];
            if (support_triple && in_str == 3) {
                if (i + 2 < src.size() && src.compare(i, 3, "\"\"\"") == 0) { out += "\"\"\""; i += 3; in_str = 0; continue; }
                out += c;
            } else if (support_triple && in_str == 4) {
                if (i + 2 < src.size() && src.compare(i, 3, "'''") == 0) { out += "'''"; i += 3; in_str = 0; continue; }
                out += c;
            } else if (in_str != 0) {
                if (escaped) { out += c; escaped = false; } 
                else if (c == '\\') { out += c; escaped = true; } 
                else if ((in_str == 1 && c == '"') || (in_str == 2 && c == '\'')) { out += c; in_str = 0; } 
                else if (c == '\n') out += "\\n"; 
                else if (c == '\r') {
                    if (i + 1 < src.size() && src[i + 1] == '\n') { out += "\\n"; ++i; } 
                    else out += "\\r";
                } 
                else out += c;
            } else {
                if (support_triple && c == '"' && i + 2 < src.size() && src.compare(i, 3, "\"\"\"") == 0) { out += "\"\"\""; i += 3; in_str = 3; continue; }
                if (support_triple && c == '\'' && i + 2 < src.size() && src.compare(i, 3, "'''") == 0) { out += "'''"; i += 3; in_str = 4; continue; }
                if (c == '"') { out += c; in_str = 1; } 
                else if (c == '\'') { out += c; in_str = 2; } 
                else out += c;
            }
            ++i;
        }
        return out;
    }

public:
    static std::string repair_cpp_string_newlines(const std::string &src) { return repair_string_newlines_impl(src, false); }
    static std::string repair_python_string_newlines(const std::string &src) { return repair_string_newlines_impl(src, true); }
    static std::string system_prompt() {
        return "You are a Senior AI Coding Agent specializing in algorithmic optimization.\n"
               "Rules:\n"
               "1. Communicate exclusively through JSON tool calls.\n"
               "2. Format: {\"tool\": \"name\", \"args\": {\"arg\": \"val\"}}\n"
               "3. Return exactly ONE JSON object. Do NOT output multiple JSON actions in a single response.\n"
               "4. Available tools (use exactly these argument names):\n"
               "   - write_file: file_path, content\n"
               "   - read_file: file_path\n"
               "   - execute_file: file_path, stdin\n"
               "   - compile_cpp: file_path\n"
               "   - run_shell_command: command\n"
               "   - make_directory: dir_path\n"
               "   - finish: (no args)\n"
               "5. For file creation tasks: write_file -> execute_file -> verify -> finish.\n"
               "6. NEVER call finish until you have EXECUTED any script that creates output files.\n"
               "7. If validation fails, read the feedback and fix the exact issue mentioned.\n"
               "8. Always implement complete, functional code without placeholders or TODO.\n"
               "9. Pay close attention to VALIDATION REQUIREMENTS.\n"
               "10. CRITICAL: Python multi-line strings MUST use triple quotes or explicit \\n escapes.\n"
               "11. CRITICAL: Inside JSON string values, escape all double quotes with backslash.\n"
               "12. CRITICAL: Write files to the EXACT paths specified in EXPECTED OUTPUT FILES.\n"
               "13. CRITICAL: For C++ JSON output, use ONLY <nlohmann/json.hpp>.\n"
               "14. CRITICAL: If a VALIDATION REQUIREMENT lists an 'expected substring', your program MUST print that exact substring to stdout during normal execution. If it would not naturally appear, add an explicit print statement, e.g. print('power') or std::cout << \"CycleError\" << std::endl;. Comments and source code do NOT count; the substring must appear in runtime output.\n"
               "15. When a [WorkspaceRecon] block is present in HISTORY, treat it as ground truth for existing files and directories; do not invent paths that do not appear there.\n";
    }
    static json build_tool_schema() {
        auto make_schema = [](const std::string &name, const json &props, const std::vector<std::string> &req) -> json {
            return {{"properties",
                     {{"tool", {{"const", name}}},
                      {"args",
                       {{"type", "object"},
                        {"properties", props},
                        {"required", req},
                        {"additionalProperties", false}}}}},
                    {"required", json::array({"tool", "args"})}};
        };
        json write_file_schema = make_schema("write_file", {{"file_path", {{"type", "string"}}}, {"content", {{"type", "string"}}}}, {"file_path", "content"});
        json read_file_schema = make_schema("read_file", {{"file_path", {{"type", "string"}}}}, {"file_path"});
        json execute_file_schema = make_schema("execute_file", {{"file_path", {{"type", "string"}}}, {"stdin", {{"type", "string"}}}}, {"file_path"});
        json compile_cpp_schema = make_schema("compile_cpp", {{"file_path", {{"type", "string"}}}}, {"file_path"});
        json run_shell_schema = make_schema("run_shell_command", {{"command", {{"type", "string"}}}}, {"command"});
        json make_dir_schema = make_schema("make_directory", {{"dir_path", {{"type", "string"}}}}, {"dir_path"});
        json finish_schema = {{"properties", {{"tool", {{"const", "finish"}}}, {"args", {{"type", "object"}, {"additionalProperties", false}}}}}, {"required", json::array({"tool", "args"})}};
        return {{"type", "object"}, {"oneOf", json::array({write_file_schema, read_file_schema, execute_file_schema, compile_cpp_schema, run_shell_schema, make_dir_schema, finish_schema})}};
    }
    static std::string get_error_hint(const std::string &error_output) {
        std::string low = error_output;
        std::transform(low.begin(), low.end(), low.begin(), ::tolower);
        if (low.find("unterminated string") != std::string::npos) return "HINT: Unterminated string literal. Use triple quotes or \\n.";
        if (low.find("indentationerror") != std::string::npos) return "HINT: Python indentation error. Use exactly 4 spaces per level.";
        if (low.find("modulenotfounderror") != std::string::npos || low.find("no module named") != std::string::npos) return "HINT: Missing Python module. Install it or use standard library.";
        if (low.find("no such file") != std::string::npos) return "HINT: File not found. Ensure write_file succeeds before execute_file.";
        if (low.find("json/json.h") != std::string::npos) return "HINT: Replace <json/json.h> with <nlohmann/json.hpp>.";
        return "";
    }
    static bool is_critical_error(const std::string &obs) {
        return obs.find("SyntaxError:") != std::string::npos ||
               obs.find("IndentationError:") != std::string::npos ||
               obs.find("ModuleNotFoundError:") != std::string::npos ||
               obs.find("No such file or directory:") != std::string::npos ||
               obs.find("fatal error:") != std::string::npos ||
               obs.find("error: expected") != std::string::npos ||
               obs.find("COMPILATION_FAILED:") != std::string::npos ||
               obs.find("invalid syntax") != std::string::npos;
    }
    static std::string error_type_from_output(const std::string &obs) {
        std::string low = obs;
        std::transform(low.begin(), low.end(), low.begin(), ::tolower);
        if (low.find("syntaxerror") != std::string::npos || low.find("invalid syntax") != std::string::npos) return "SyntaxError";
        if (low.find("indentationerror") != std::string::npos) return "IndentationError";
        if (low.find("modulenotfound") != std::string::npos) return "ModuleNotFound";
        if (low.find("nameerror") != std::string::npos) return "NameError";
        if (low.find("typeerror") != std::string::npos) return "TypeError";
        if (low.find("compilation_failed") != std::string::npos || low.find("fatal error:") != std::string::npos) return "CompilationError";
        if (low.find("no such file") != std::string::npos) return "FileNotFound";
        return "RuntimeError";
    }
    static std::string validation_cache_key(const json &check, const std::string &fp) {
        std::string input = check.contains("input") && check["input"].is_string() ? check["input"].get<std::string>() : "";
        std::string expect = check.contains("expect") && check["expect"].is_string() ? check["expect"].get<std::string>() : "";
        return fp + "|" + input + "|" + expect;
    }
};

struct EpisodeMetrics {
    bool success = false;
    int turns_used = 0;
    int max_turns = 6;
    int llm_calls = 0;
    int tool_calls = 0;
    int duplicate_actions_blocked = 0;
    int critical_errors = 0;
    int non_zero_executions = 0;
    int missing_expected_files = 0;
    int validation_failures = 0;
    int memory_hits = 0;
    int api_calls_saved = 0;
    double plan_score = 0.0;
    double wall_time_sec = 0.0;
};

inline double compute_reward(const EpisodeMetrics& m) {
    double r = m.success ? 10.0 : 0.0;
    if (m.success) {
        double turn_efficiency = 1.0 - static_cast<double>(m.turns_used) / m.max_turns;
        double call_efficiency = 1.0 / (1.0 + 0.15 * m.llm_calls);
        r += 3.0 * turn_efficiency + 2.0 * call_efficiency;
    } else {
        r -= 1.5 * (static_cast<double>(m.turns_used) / m.max_turns);
    }
    double progress = 0.0;
    progress += 0.8 * (1.0 - std::min(1.0, m.missing_expected_files / 5.0));
    progress += 0.6 * m.plan_score;
    progress -= 0.4 * m.validation_failures;
    progress -= 0.3 * m.non_zero_executions;
    r += std::clamp(progress, -2.0, 3.0);
    r -= 1.5 * m.critical_errors;
    r -= 0.4 * m.duplicate_actions_blocked;
    if (m.memory_hits > 0 && m.success)
        r += 1.5 + 0.3 * std::min(m.api_calls_saved, 5);
    else if (m.memory_hits > 0 && !m.success)
        r -= 0.5;
    if (m.wall_time_sec > 120.0)
        r -= 0.01 * (m.wall_time_sec - 120.0);
    return std::clamp(r, -8.0, 18.0);
}

class AgentCritic {
public:
    struct ValidationResult {
        bool passed;
        std::string output;
        std::string error;
    };
    AgentCritic(ToolExecutor& te, SemanticMemoryManager& sm)
        : tool_executor_(te), semantic_memory_(sm) {}
    void reset() {
        expected_outputs_.clear();
        validation_checks_.clear();
        current_task_.clear();
        executed_action_states_.clear();
        executed_action_history_.clear();
        semantic_action_counts_.clear();
        recent_semantics_.clear();
        recent_action_rewards_.clear();
        validation_cache_.clear();
        recovery_execution_cache_.clear();
        extended_check_cache_.clear();
        written_files_.clear();
        last_progress_ = -1.0;
        no_progress_turns_ = 0;
        blocked_action_count_ = 0;
        recovery_actions_count_ = 0;
        finish_called_prematurely_ = false;
        llm_calls_observed_ = 0;
        min_validation_failures_ = 999;
        last_write_attempts_.clear();
        file_block_counts_.clear();
    }
    void set_context(const std::vector<std::string>& expected,
                     const std::vector<json>& checks,
                     const std::string& task) {
        expected_outputs_ = expected;
        validation_checks_ = checks;
        current_task_ = task;
    }
    const std::vector<std::string>& written_files() const { return written_files_; }
    std::vector<std::string>& written_files() { return written_files_; }
    int& blocked_action_count() { return blocked_action_count_; }
    int no_progress_turns() const { return no_progress_turns_; }
    int min_validation_failures() const { return min_validation_failures_; }
    void update_min_failures(int n) {
        if (n < min_validation_failures_) min_validation_failures_ = n;
    }
    void set_llm_call_count(int n) { llm_calls_observed_ = n; }
    void invalidate_validation_cache_for(const std::string& /*fp*/) {
        validation_cache_.clear();
        recovery_execution_cache_.clear();
        extended_check_cache_.clear();
    }
    bool all_validations_pass() const {
        if (!expected_outputs_.empty()) {
            for (const auto& f : expected_outputs_)
                if (!fs::exists(f)) return false;
        }
        for (const auto& c : validation_checks_)
            if (!passes_check(c)) return false;
        return true;
    }
    std::vector<std::string> collect_validation_failures() const {
        std::vector<std::string> fails;
        for (const auto& f : expected_outputs_) {
            if (!fs::exists(f))
                fails.push_back("Missing expected file: " + f);
        }
        for (const auto& c : validation_checks_) {
            if (!passes_check(c)) {
                std::string desc = c.value("type", "check");
                if (c.contains("file")) desc += " on " + c.value("file", "");
                if (c.contains("expect")) desc += " expecting '" + c.value("expect", "") + "'";
                fails.push_back("Validation failed: " + desc);
            }
        }
        return fails;
    }
    std::string execute_cached(const std::string& fp, const std::string& stdin_input) const {
        const std::string key = fp + "\n" + stdin_input;
        auto it = recovery_execution_cache_.find(key);
        if (it != recovery_execution_cache_.end()) return it->second;
        std::string out = tool_executor_.executor.execute_file(fp, stdin_input);
        recovery_execution_cache_[key] = out;
        return out;
    }
    ValidationResult get_or_execute_validation(const json& check, const std::string& fp) const {
        const std::string key = check.dump() + "|" + fp;
        auto it = validation_cache_.find(key);
        if (it != validation_cache_.end()) return it->second;
        ValidationResult vr;
        vr.passed = passes_check(check);
        if (check.value("type", "") == "execution") {
            vr.output = execute_cached(fp, check.value("input", ""));
        }
        validation_cache_[key] = vr;
        return vr;
    }
    bool passes_check(const json& check) const {
        const std::string t = check.value("type", "");
        if (t == "file_exists") {
            return fs::exists(check.value("path", ""));
        }
        if (t == "execution") {
            const std::string fp = check.value("file", "");
            if (!fs::exists(fp)) return false;
            const std::string out = execute_cached(fp, check.value("input", ""));
            if (out.find("Return code: 0") == std::string::npos) return false;
            if (check.contains("expect")) {
                std::string expect = check.value("expect", "");
                std::string low_out = out, low_exp = expect;
                std::transform(low_out.begin(), low_out.end(), low_out.begin(), ::tolower);
                std::transform(low_exp.begin(), low_exp.end(), low_exp.begin(), ::tolower);
                if (low_out.find(low_exp) == std::string::npos) return false;
            }
            return true;
        }
        if (t == "perf_measure") return passes_perf_measure(check);
        if (t == "import_scan") return passes_import_scan(check);
        if (t == "ast_diff") return passes_ast_diff(check);
        if (t == "nondeterministic") return passes_nondeterministic(check);
        if (t == "rubric") return passes_rubric(check);
        if (t == "behavior_probe") return passes_behavior_probe(check);
        return true;
    }
    std::string get_action_target_file(const json& action) const {
        const std::string tool = action.value("tool", "");
        const json args = action.value("args", json::object());
        if (tool == "write_file" || tool == "read_file" || tool == "execute_file" ||
            tool == "compile_cpp") {
            return args.value("file_path", "");
        }
        if (tool == "run_shell_command") {
            return args.value("command", "").substr(0, 64);
        }
        return "";
    }
    std::string semantic_fingerprint(const json& action) const {
        const std::string tool = action.value("tool", "");
        if (tool.empty()) return "";
        std::string fp = get_action_target_file(action);
        if (tool == "write_file") {
            const std::string content = action.value("args", json::object()).value("content", "");
            return tool + ":" + fp + ":" + md5_hash(content).substr(0, 8);
        }
        return tool + ":" + fp;
    }
    std::string fingerprint_action_stateful(const json& action) const {
        const std::string base = semantic_fingerprint(action);
        const std::string target = get_action_target_file(action);
        if (target.empty()) return base;
        return base + "|state:" + CodingAgentUtils::validation_file_signature(target);
    }
    bool detect_ping_pong(const std::string& candidate) const {
        if (no_progress_turns_ < 1) return false;
        const size_t sz = recent_semantics_.size();
        if (sz < 2) return false;
        return recent_semantics_[sz - 2] == candidate && recent_semantics_[sz - 1] != candidate;
    }
    bool should_skip_duplicate_action(const json& action) {
        const std::string tool = action.value("tool", "");
        if (tool == "finish") return false;
        const std::string sem = semantic_fingerprint(action);
        if (sem.empty()) return false;
        if (tool == "write_file") {
            const json args        = action.value("args", json::object());
            const std::string fp      = args.value("file_path", "");
            const std::string content = args.value("content", "");
            if (!fp.empty() && !content.empty() &&
                is_fuzzy_duplicate_write(fp, content)) {
                file_block_counts_[fp]++;
                return true;
            }
            if (!fp.empty() && fs::exists(fp)) {
                if (md5_hash(read_file_text(fp)) == md5_hash(content)) {
                    file_block_counts_[fp]++;
                    return true;
                }
            }
        }
        if (auto it = semantic_action_counts_.find(sem);
            it != semantic_action_counts_.end() && it->second >= 2) {
            return true;
        }
        if (!executed_action_history_.empty() &&
            executed_action_history_.back() == fingerprint_action_stateful(action)) {
            return true;
        }
        if (tool == "read_file" || tool == "execute_file" || tool == "compile_cpp") {
            const std::string target = get_action_target_file(action);
            auto st = executed_action_states_.find(target);
            if (st != executed_action_states_.end() &&
                st->second == CodingAgentUtils::validation_file_signature(target) &&
                semantic_action_counts_.count(sem)) {
                return true;
            }
        }
        return detect_ping_pong(sem);
    }
    void record_executed_action(const json& action) {
        const std::string sem = semantic_fingerprint(action);
        if (sem.empty()) return;
        semantic_action_counts_[sem]++;
        recent_semantics_.push_back(sem);
        if (recent_semantics_.size() > 16)
            recent_semantics_.erase(recent_semantics_.begin());
        executed_action_history_.push_back(fingerprint_action_stateful(action));
        const std::string target = get_action_target_file(action);
        if (!target.empty())
            executed_action_states_[target] = CodingAgentUtils::validation_file_signature(target);
        if (action.value("tool", "") == "write_file") {
            const json args = action.value("args", json::object());
            const std::string fp = args.value("file_path", "");
            const std::string content = args.value("content", "");
            if (!fp.empty() && !content.empty() && fs::exists(fp)) {
                record_write_attempt(fp, content);
            }
        }
    }
    double progress_ratio() const {
        if (expected_outputs_.empty() && validation_checks_.empty()) return 1.0;
        int total = static_cast<int>(expected_outputs_.size() + validation_checks_.size());
        if (total == 0) return 1.0;
        int passed = 0;
        for (const auto& f : expected_outputs_)
            if (fs::exists(f)) ++passed;
        for (const auto& check : validation_checks_)
            if (passes_check(check)) ++passed;
        return static_cast<double>(passed) / total;
    }
    void update_progress() {
        double p = progress_ratio();
        if (p > last_progress_ + 1e-6) { last_progress_ = p; no_progress_turns_ = 0; }
        else ++no_progress_turns_;
    }
    json synthesize_recovery_action() {
        for (const auto& c : validation_checks_) {
            if (c.value("type", "") != "execution") continue;
            const std::string fp = c.value("file", "");
            if (!fs::exists(fp)) continue;
            ValidationResult vr = get_or_execute_validation(c, fp);
            if (!vr.passed) {
                json act = {{"tool", "execute_file"},
                            {"args", {{"file_path", fp}, {"stdin", c.value("input", "")}}}};
                if (!should_skip_duplicate_action(act)) return act;
            }
        }
        for (const auto& f : expected_outputs_) {
            if (!fs::exists(f)) continue;
            const std::string ext = fs::path(f).extension().string();
            if (ext != ".cpp" && ext != ".cc" && ext != ".cxx") continue;
            json act = {{"tool", "compile_cpp"}, {"args", {{"file_path", f}}}};
            if (!should_skip_duplicate_action(act)) return act;
        }
        for (const auto& f : expected_outputs_) {
            if (!fs::exists(f)) continue;
            json act = {{"tool", "read_file"}, {"args", {{"file_path", f}}}};
            if (!should_skip_duplicate_action(act)) return act;
        }
        std::string stuck_target;
        for (auto it = recent_semantics_.rbegin(); it != recent_semantics_.rend(); ++it) {
            const std::string& sem = *it;
            size_t first_colon = sem.find(':');
            if (first_colon != std::string::npos) {
                size_t second_colon = sem.find(':', first_colon + 1);
                std::string target = (second_colon != std::string::npos)
                                         ? sem.substr(first_colon + 1, second_colon - first_colon - 1)
                                         : sem.substr(first_colon + 1);
                while (!target.empty() && target.back() == ' ') target.pop_back();
                if (!target.empty() && fs::exists(target)) { stuck_target = target; break; }
            }
        }
        if (!stuck_target.empty()) {
            json act = {{"tool", "read_file"}, {"args", {{"file_path", stuck_target}}}};
            if (!should_skip_duplicate_action(act)) return act;
        }
        return json::object();
    }
    void record_trajectory_rewards(const std::vector<json>& actions, double final_reward) {
        for (const auto& act : actions) {
            std::string sem = semantic_fingerprint(act);
            recent_action_rewards_.push_back({sem, final_reward});
        }
        if (recent_action_rewards_.size() > 20)
            recent_action_rewards_.erase(recent_action_rewards_.begin());
    }
    std::string build_action_penalty_hint() const {
        std::string hint;
        std::unordered_map<std::string, std::pair<double, int>> sem_stats;
        for (const auto& [sem, rew] : recent_action_rewards_) {
            sem_stats[sem].first += rew;
            sem_stats[sem].second++;
        }
        for (const auto& [sem, stats] : sem_stats) {
            double avg_rew = stats.first / stats.second;
            if (avg_rew < -2.0 && stats.second >= 2) {
                hint += "WARNING: The action '" + sem +
                        "' has a negative historical reward. "
                        "DO NOT repeat this exact action unless the file state has fundamentally "
                        "changed.\n";
            }
        }
        return hint;
    }
    json generate_structured_critique(bool success, int turns_used,
                                      const std::vector<json>& actions,
                                      const EpisodeMetrics& metrics) const {
        json critique;
        critique["success"] = success;
        critique["turns_used"] = turns_used;
        critique["max_turns"] = metrics.max_turns;
        critique["tool_calls"] = static_cast<int>(actions.size());
        critique["blocked_duplicates"] = metrics.duplicate_actions_blocked;
        critique["no_progress_turns"] = no_progress_turns_;
        critique["recovery_actions"] = recovery_actions_count_;
        critique["progress_ratio"] = progress_ratio();
        json what_worked = json::array();
        if (success) {
            what_worked.push_back("All expected outputs and validation checks ultimately passed.");
            if (turns_used <= 3)
                what_worked.push_back("Efficient trajectory: solved within " +
                                      std::to_string(turns_used) + " turns.");
            if (metrics.duplicate_actions_blocked == 0)
                what_worked.push_back("No redundant actions were attempted.");
            if (metrics.memory_hits > 0)
                what_worked.push_back("Memory retrieval contributed useful prior experience.");
            if (recovery_actions_count_ == 0)
                what_worked.push_back("No recovery actions were required; plan was sufficiently robust.");
        } else {
            if (progress_ratio() > 0.5)
                what_worked.push_back("Partial progress was achieved before failure.");
        }
        critique["what_worked"] = what_worked;
        json wasteful = json::array();
        if (metrics.duplicate_actions_blocked > 0)
            wasteful.push_back("Blocked " + std::to_string(metrics.duplicate_actions_blocked) +
                               " redundant actions.");
        if (no_progress_turns_ >= 2)
            wasteful.push_back("Stalled for " + std::to_string(no_progress_turns_) +
                               " consecutive turns with no validation progress.");
        if (recovery_actions_count_ > 2)
            wasteful.push_back("Required multiple recovery interventions.");
        critique["wasteful_patterns"] = wasteful;
        json recommendations = json::array();
        if (!success) {
            recommendations.push_back("Prioritize minimal structural edits over full rewrites once an artifact exists.");
            if (no_progress_turns_ >= 2)
                recommendations.push_back("Escalate to deterministic recovery earlier when progress stalls.");
            if (metrics.duplicate_actions_blocked > 3)
                recommendations.push_back("After repeated blocks, allow modest content divergence under the same failure signature.");
        }
        critique["recommendations"] = recommendations;
        return critique;
    }
    void compute_and_store_metrics(bool success, int turns_used, int max_turns,
                                   const std::string& task,
                                   const std::vector<json>& actions) {
        EpisodeMetrics m;
        m.success = success;
        m.turns_used = turns_used;
        m.max_turns = max_turns;
        m.duplicate_actions_blocked = blocked_action_count_;
        m.llm_calls = llm_calls_observed_;
        m.memory_hits = 0;
        m.validation_failures = min_validation_failures_ == 999 ? 0 : min_validation_failures_;
        (void)generate_structured_critique(success, turns_used, actions, m);
        (void)task;
    }
private:
    struct PerfSample {
        bool spawned = false;
        bool exited_ok = false;
        int exit_code = -1;
        double wall_seconds = 0.0;
        long peak_rss_kb = 0;
        std::string captured;
    };
    PerfSample run_perf_sample(const std::string& cmd, double timeout_s) const {
        PerfSample s;
        int pipefd[2];
        if (::pipe(pipefd) != 0) return s;
        pid_t pid = ::fork();
        if (pid < 0) {
            ::close(pipefd[0]);
            ::close(pipefd[1]);
            return s;
        }
        if (pid == 0) {
            ::close(pipefd[0]);
            ::dup2(pipefd[1], STDOUT_FILENO);
            ::dup2(pipefd[1], STDERR_FILENO);
            ::close(pipefd[1]);
            ::setpgid(0, 0);
            execl("/bin/sh", "sh", "-c", cmd.c_str(), (char*)nullptr);
            _exit(127);
        }
        ::close(pipefd[1]);
        s.spawned = true;
        const auto start = std::chrono::steady_clock::now();
        const auto deadline = start + std::chrono::duration<double>(timeout_s);
        char buf[4096];
        const size_t capture_limit = 65536;
        bool killed = false;
        while (true) {
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline && !killed) {
                ::kill(-pid, SIGKILL);
                killed = true;
            }
            struct pollfd pfd{pipefd[0], POLLIN, 0};
            long long remain_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                      deadline - now).count();
            int timeout_ms = static_cast<int>(
                std::max<long long>(0, std::min<long long>(remain_ms, 200)));
            int pr = ::poll(&pfd, 1, timeout_ms);
            if (pr > 0) {
                ssize_t n = ::read(pipefd[0], buf, sizeof(buf));
                if (n > 0) {
                    if (s.captured.size() < capture_limit) {
                        size_t take = std::min<size_t>(
                            static_cast<size_t>(n), capture_limit - s.captured.size());
                        s.captured.append(buf, take);
                    }
                } else if (n == 0) {
                    break;
                }
            }
            if (killed && pr <= 0) break;
        }
        ::close(pipefd[0]);
        int status = 0;
        struct rusage ru{};
        ::wait4(pid, &status, 0, &ru);
        s.wall_seconds = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - start).count();
        s.peak_rss_kb  = ru.ru_maxrss;
        s.exit_code    = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
        s.exited_ok    = !killed && WIFEXITED(status) && WEXITSTATUS(status) == 0;
        return s;
    }
    bool passes_perf_measure(const json& check) const {
        std::string target;
        for (const auto& f : expected_outputs_) {
            const std::string ext = fs::path(f).extension().string();
            if ((ext == ".py" || ext == ".cpp" || ext == ".cc" ||
                 ext == ".cxx" || ext == ".sh") && fs::exists(f)) {
                target = f;
                break;
            }
        }
        if (target.empty()) {
            std::cerr << "[AgentCritic] perf_measure: no runnable target in "
                         "expected_outputs — lenient pass.\n";
            return true;
        }
        const double wall_max = check.value("wall_time_max_s", 1e9);
        const long   rss_max_kb = static_cast<long>(
            check.value("peak_rss_max_mb", 1e9) * 1024.0);
        const std::string ext = fs::path(target).extension().string();
        std::string cmd;
        if (ext == ".py") {
            cmd = "python3 \"" + target + "\"";
        } else if (ext == ".sh") {
            cmd = "bash \"" + target + "\"";
        } else {
            std::string exe = tool_executor_.executor.compile_cpp(target);
            if (exe.find("COMPILATION_FAILED:") != std::string::npos) {
                std::cerr << "[AgentCritic] perf_measure: compilation failed for "
                          << target << "\n";
                return false;
            }
            cmd = "\"" + exe + "\"";
        }
        const double timeout_s = std::max(30.0, wall_max + 5.0);
        PerfSample s = run_perf_sample(cmd, timeout_s);
        if (!s.spawned) {
            std::cerr << "[AgentCritic] perf_measure: spawn failed for "
                      << target << "\n";
            return false;
        }
        if (!s.exited_ok) {
            std::cerr << "[AgentCritic] perf_measure FAILED: " << target
                      << " did not exit 0 (exit=" << s.exit_code
                      << ", wall=" << s.wall_seconds << "s)\n"
                      << s.captured.substr(0, 400) << "\n";
            return false;
        }
        if (s.wall_seconds > wall_max) {
            std::cerr << "[AgentCritic] perf_measure FAILED: " << s.wall_seconds
                      << "s > " << wall_max << "s for " << target << "\n";
            return false;
        }
        if (check.contains("peak_rss_max_mb")) {
            if (s.peak_rss_kb > rss_max_kb) {
                std::cerr << "[AgentCritic] perf_measure FAILED: peak RSS "
                          << (s.peak_rss_kb / 1024.0) << " MB > "
                          << (rss_max_kb / 1024.0) << " MB for " << target << "\n";
                return false;
            }
        }
        if (check.contains("speedup_min")) {
            if (!check.contains("baseline_cmd") || !check["baseline_cmd"].is_string()) {
                std::cerr << "[AgentCritic] perf_measure CONFIG ERROR: "
                             "speedup_min requested but baseline_cmd is absent. "
                             "This is a test-definition bug, not an agent bug — "
                             "failing closed.\n";
                return false;
            }
            const std::string base_cmd = check["baseline_cmd"].get<std::string>();
            PerfSample base = run_perf_sample(base_cmd, timeout_s);
            if (!base.exited_ok) {
                std::cerr << "[AgentCritic] perf_measure: baseline_cmd failed "
                             "(exit=" << base.exit_code << "): "
                          << base.captured.substr(0, 200) << "\n";
                return false;
            }
            const double speedup = base.wall_seconds / std::max(1e-9, s.wall_seconds);
            const double need    = check.value("speedup_min", 1.0);
            if (speedup < need) {
                std::cerr << "[AgentCritic] perf_measure FAILED: speedup "
                          << speedup << "x < " << need
                          << "x (baseline=" << base.wall_seconds
                          << "s, candidate=" << s.wall_seconds << "s)\n";
                return false;
            }
        }
        if (check.contains("max_abs_error")) {
            if (!check.contains("reference_file") ||
                !check["reference_file"].is_string()) {
                std::cerr << "[AgentCritic] perf_measure CONFIG ERROR: "
                             "max_abs_error requested but reference_file is absent. "
                             "Failing closed.\n";
                return false;
            }
            const std::string ref_path = check["reference_file"].get<std::string>();
            if (!fs::exists(ref_path)) {
                std::cerr << "[AgentCritic] perf_measure FAILED: reference_file "
                             "not found: " << ref_path << "\n";
                return false;
            }
            std::ifstream rf(ref_path);
            std::istringstream cand(s.captured);
            std::string ref_line, cand_line;
            const double tol = check.value("max_abs_error", 0.0);
            bool any_compared = false;
            while (std::getline(rf, ref_line)) {
                if (!std::getline(cand, cand_line)) {
                    std::cerr << "[AgentCritic] perf_measure FAILED: candidate "
                                 "output shorter than reference\n";
                    return false;
                }
                std::istringstream rss(ref_line), css(cand_line);
                double rv = 0.0, cv = 0.0;
                if (rss >> rv) {
                    if (!(css >> cv)) {
                        std::cerr << "[AgentCritic] perf_measure FAILED: candidate "
                                     "line is not numeric: " << cand_line << "\n";
                        return false;
                    }
                    any_compared = true;
                    if (std::fabs(rv - cv) > tol) {
                        std::cerr << "[AgentCritic] perf_measure FAILED: |"
                                  << rv << " - " << cv << "| > " << tol << "\n";
                        return false;
                    }
                }
            }
            if (!any_compared) {
                std::cerr << "[AgentCritic] perf_measure FAILED: no numeric lines "
                             "were compared against reference\n";
                return false;
            }
        }
        return true;
    }
    bool passes_import_scan(const json& check) const {
        if (check.contains("forbid") && check["forbid"].is_array()) {
            for (const auto& f : expected_outputs_) {
                if (!fs::exists(f)) continue;
                const std::string content = read_file_text(f);
                for (const auto& item : check["forbid"]) {
                    if (!item.is_string()) continue;
                    if (contains_import_usage(content, item.get<std::string>()))
                        return false;
                }
            }
        }
        if (check.contains("require") && check["require"].is_string()) {
            const std::string req = check["require"].get<std::string>();
            bool found = false;
            for (const char* manifest : {"requirements.txt", "pyproject.toml", "setup.py", "setup.cfg"}) {
                if (!fs::exists(manifest)) continue;
                if (read_file_text(manifest).find(req) != std::string::npos) { found = true; break; }
            }
            if (!found) return false;
        }
        return true;
    }
    static bool contains_import_usage(const std::string& src, const std::string& name) {
        static const std::unordered_set<std::string> stdlib_names = {
            "json", "ast", "re", "os", "sys", "math", "random", "collections",
            "itertools", "functools", "heapq", "queue", "threading", "time",
            "datetime", "csv", "sqlite3", "hashlib", "concurrent", "typing"
        };
        std::string esc;
        for (char c : name) {
            if (std::strchr("\\.^$|()[]{}*+?", c)) esc += '\\';
            esc += c;
        }
        try {
            std::regex import_re("\\b(import\\s+" + esc + "\\b|from\\s+" + esc + "\\b)");
            if (std::regex_search(src, import_re)) return true;
            std::regex call_re("\\b" + esc + "\\s*\\(");
            if (std::regex_search(src, call_re)) {
                if (name == "eval" || name == "exec") return true;
                if (stdlib_names.count(name) == 0) return true;
            }
        } catch (const std::regex_error&) {
            return src.find(name) != std::string::npos;
        }
        return false;
    }
    bool passes_ast_diff(const json& check) const {
        if (check.contains("allowed_files") && check["allowed_files"].is_array()) {
            std::set<std::string> allowed;
            for (const auto& f : check["allowed_files"])
                if (f.is_string()) allowed.insert(f.get<std::string>());
            for (const auto& f : written_files_) {
                if (!allowed.count(f))
                    std::cerr << "[AgentCritic] ast_diff: wrote outside allowed set: "
                              << f << " (lenient — not failing)\n";
            }
        }
        return true;
    }
    bool passes_nondeterministic(const json& check) const {
        if (!check.contains("expected_final")) return true;
        std::string target;
        for (const auto& f : expected_outputs_) {
            if (fs::path(f).extension() == ".py" && fs::exists(f)) { target = f; break; }
        }
        if (target.empty()) {
            std::cerr << "[AgentCritic] nondeterministic: no .py target in expected_outputs "
                         "— lenient pass.\n";
            return true;
        }
        const int trials   = check.value("trials", 5);
        const int expected = check.value("expected_final", 0);
        const bool perfect = check.value("require_perfect", true);
        const int cap      = std::min(trials, 10);
        int passes = 0;
        for (int i = 0; i < cap; ++i) {
            const std::string out = tool_executor_.executor.execute_file(target);
            if (out.find("Return code: 0") != std::string::npos &&
                out.find(std::to_string(expected)) != std::string::npos)
                ++passes;
        }
        const bool ok = perfect ? (passes == cap) : (passes > 0);
        if (!ok)
            std::cerr << "[AgentCritic] nondeterministic: " << passes << "/" << cap
                      << " trials passed for " << target << "\n";
        return ok;
    }
    bool passes_rubric(const json& /*check*/) const {
        std::cerr << "[AgentCritic] rubric check present but no grader configured — "
                     "lenient pass.\n";
        return true;
    }
    bool passes_behavior_probe(const json& check) const {
        if (check.contains("forbid_paths") && check["forbid_paths"].is_array()) {
            for (const auto& f : written_files_) {
                if (!fs::exists(f)) continue;
                const std::string content = read_file_text(f);
                for (const auto& p : check["forbid_paths"]) {
                    if (!p.is_string()) continue;
                    if (content.find(p.get<std::string>()) != std::string::npos)
                        return false;
                }
            }
        }
        return true;
    }
    static std::string read_file_text(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) return {};
        std::ostringstream ss;
        ss << in.rdbuf();
        return ss.str();
    }
    struct WriteAttempt {
        std::string content_hash;
        std::vector<std::string> normalized_lines;
        std::string failure_signature;
        double timestamp = 0.0;
    };
    static constexpr size_t kWriteHistoryDepth = 4;
    static constexpr double kJaccardDefault = 0.85;
    static constexpr double kJaccardEscalated = 1.0;
    static constexpr int kEscalateAfterBlocks = 2;
    static std::vector<std::string> normalize_content_lines(const std::string& src) {
        std::vector<std::string> out;
        out.reserve(src.size() / 24);
        std::string line;
        line.reserve(128);
        auto flush = [&]() {
            size_t a = line.find_first_not_of(" \t\r");
            size_t b = line.find_last_not_of(" \t\r");
            if (a != std::string::npos && b != std::string::npos) {
                std::string c;
                c.reserve(b - a + 1);
                bool in_ws = false;
                for (size_t i = a; i <= b; ++i) {
                    char ch = line[i];
                    if (ch == ' ' || ch == '\t') {
                        if (!in_ws) { c += ' '; in_ws = true; }
                    } else { c += ch; in_ws = false; }
                }
                out.push_back(std::move(c));
            }
            line.clear();
        };
        for (char c : src) {
            if (c == '\n') flush();
            else line += c;
        }
        if (!line.empty()) flush();
        return out;
    }
    static double line_jaccard(const std::vector<std::string>& a,
                                const std::vector<std::string>& b) {
        if (a.empty() || b.empty()) return 0.0;
        std::unordered_set<std::string> sa(a.begin(), a.end());
        std::unordered_set<std::string> sb(b.begin(), b.end());
        size_t inter = 0;
        for (const auto& s : sa) if (sb.count(s)) ++inter;
        size_t uni = sa.size() + sb.size() - inter;
        return uni ? static_cast<double>(inter) / uni : 1.0;
    }
    static std::string normalized_content_hash(const std::string& src) {
        auto lines = normalize_content_lines(src);
        std::string joined;
        joined.reserve(src.size());
        for (auto& l : lines) { joined += l; joined += '\n'; }
        return md5_hash(joined);
    }
    std::string validation_failure_signature() const {
        auto fails = collect_validation_failures();
        if (fails.empty()) return "PASS";
        std::string sig;
        for (const auto& f : fails) {
            sig += f.substr(0, std::min<size_t>(64, f.size()));
            sig += '|';
        }
        return md5_hash(sig);
    }
    bool has_verifiable_validation() const {
        if (!expected_outputs_.empty()) return true;
        for (const auto& c : validation_checks_) {
            const std::string t = c.value("type", "");
            if (t == "file_exists" || t == "execution") return true;
        }
        return false;
    }
    double dedup_threshold_for(const std::string& fp) const {
        auto it = file_block_counts_.find(fp);
        if (it != file_block_counts_.end() && it->second >= kEscalateAfterBlocks)
            return kJaccardEscalated;
        return kJaccardDefault;
    }
    bool is_fuzzy_duplicate_write(const std::string& fp, const std::string& content) const {
        if (!has_verifiable_validation()) return false;
        auto it = last_write_attempts_.find(fp);
        if (it == last_write_attempts_.end() || it->second.empty()) return false;
        const std::string sig = validation_failure_signature();
        const std::string chash = normalized_content_hash(content);
        for (const auto& attempt : it->second) {
            if (attempt.failure_signature != sig) continue;
            if (attempt.content_hash == chash) return true;
        }
        return false;
    }
    void record_write_attempt(const std::string& fp, const std::string& content) {
        auto& hist = last_write_attempts_[fp];
        WriteAttempt a;
        a.content_hash = normalized_content_hash(content);
        a.normalized_lines = normalize_content_lines(content);
        a.failure_signature = validation_failure_signature();
        a.timestamp = get_time();
        hist.push_back(std::move(a));
        if (hist.size() > kWriteHistoryDepth) hist.erase(hist.begin());
    }
    ToolExecutor& tool_executor_;
    SemanticMemoryManager& semantic_memory_;
    std::vector<std::string> expected_outputs_;
    std::vector<json> validation_checks_;
    std::string current_task_;
    std::unordered_map<std::string, std::string> executed_action_states_;
    std::vector<std::string> executed_action_history_;
    std::unordered_map<std::string, int> semantic_action_counts_;
    std::vector<std::string> recent_semantics_;
    std::vector<std::pair<std::string, double>> recent_action_rewards_;
    mutable std::unordered_map<std::string, ValidationResult> validation_cache_;
    mutable std::unordered_map<std::string, std::string> recovery_execution_cache_;
    mutable std::unordered_map<std::string, bool> extended_check_cache_;
    std::vector<std::string> written_files_;
    double last_progress_ = -1.0;
    int no_progress_turns_ = 0;
    int blocked_action_count_ = 0;
    int recovery_actions_count_ = 0;
    bool finish_called_prematurely_ = false;
    int llm_calls_observed_ = 0;
    int min_validation_failures_ = 999;
    std::unordered_map<std::string, std::vector<WriteAttempt>> last_write_attempts_;
    std::unordered_map<std::string, int> file_block_counts_;
};

class AgentPlanner {
public:
    AgentPlanner(LLMBackendManager& llm, TaskDecomposer& decomp, DependencyPlanner& dep)
        : llm_(llm), decomposer_(decomp), dep_planner_(dep) {}
    std::vector<json> generate_validation_conditioned_plan(const std::string& task,
                                                           const std::vector<std::string>& expected_outputs,
                                                           const std::vector<json>& validation_checks) {
        return dep_planner_.generate_plan(task, expected_outputs, validation_checks);
    }
    std::pair<double, std::vector<std::string>> score_plan_against_validation(
        const std::vector<json>& plan,
        const std::vector<std::string>& expected_outputs,
        const std::vector<json>& validation_checks) {
        std::set<std::string> plan_files;
        for (const auto& a : plan) {
            if (a.value("tool", "") == "write_file") {
                plan_files.insert(a.value("args", json::object()).value("file_path", ""));
            }
        }
        int matched = 0;
        std::vector<std::string> missing;
        for (const auto& f : expected_outputs) {
            if (plan_files.count(f))
                ++matched;
            else
                missing.push_back(f);
        }
        double file_score =
            expected_outputs.empty() ? 1.0 : static_cast<double>(matched) / expected_outputs.size();
        double content_score = validation_checks.empty() ? 1.0 : 0.8;
        double score = 0.6 * std::max(0.0, file_score) + 0.4 * content_score;
        return {score, missing};
    }
    std::vector<json> repair_plan_suffix(const std::vector<json>& base_plan,
                                         const std::vector<std::string>& missing_files,
                                         const std::string& task,
                                         const std::string& validation_reqs) {
        std::string missing_desc =
            missing_files.empty() ? "" : ("Missing target files:\n" + join(missing_files, ", ") + "\n");
        const std::string prompt = "Task: " + task + "\n" + validation_reqs + "\n" + missing_desc +
                                   "Generate ONLY the additional tool actions needed to write these files and satisfy all validation checks.";
        const json schema = {
            {"type", "object"},
            {"properties",
             {{"actions",
               {{"type", "array"},
                {"items",
                 {{"type", "object"},
                  {"properties",
                   {{"tool", {{"type", "string"}}},
                    {"args",
                     {{"type", "object"},
                      {"properties",
                       {{"file_path", {{"type", "string"}}}, {"content", {{"type", "string"}}}}},
                      {"required", json::array({"file_path", "content"})}}}}},
                  {"required", json::array({"tool", "args"})}}}}}}},
            {"required", json::array({"actions"})}};
        json resp = llm_.structured_chat({{"user", prompt}}, schema, 0.0, 4096);
        std::vector<json> repaired = base_plan;
        if (resp.is_object() && resp.contains("actions") && resp["actions"].is_array()) {
            for (const auto& a : resp["actions"])
                repaired.push_back(a);
        } else if (resp.is_array()) {
            for (const auto& a : resp)
                repaired.push_back(a);
        }
        return repaired;
    }
    std::vector<json> extract_replayable_plan(const std::string& cached) {
        std::vector<json> plan;
        try {
            json j = json::parse(cached);
            json plan_node;
            if (j.is_object() && j.contains("plan")) {
                plan_node = j["plan"];                // current stored format
            } else if (j.is_array()) {
                plan_node = j;                        // legacy / direct-array format
            }
            if (plan_node.is_array()) {
                for (const auto& a : plan_node) {
                    if (a.is_object() && a.contains("tool"))
                        plan.push_back(a);
                }
            }
        } catch (...) {
        }
        return plan;
    }    
    std::vector<std::string> decompose_plan(const std::vector<json>& plan) {
        return decomposer_.decompose_plan(plan);
    }

private:
    LLMBackendManager& llm_;
    TaskDecomposer& decomposer_;
    DependencyPlanner& dep_planner_;
    static std::string join(const std::vector<std::string>& vec, const std::string& delim) {
        std::string result;
        for (size_t i = 0; i < vec.size(); ++i) {
            result += vec[i];
            if (i + 1 < vec.size())
                result += delim;
        }
        return result;
    }
};

class CodingAgent {
public:
    CodingAgent()
        : llm(LLMBackendManager::Config{
              {{"google", "gemini-3.8-flash"},
               {"openrouter", "thinkingmachines/inkling-small:free"},
               {"groq", "openai/gpt-oss-120b"}},
              {"google", "openrouter", "groq"},
              30}),
          decomposer(llm),
          planner_dep(llm),
          critic(tool_executor, semantic_memory),
          agent_planner(llm, decomposer, planner_dep) {}
    std::vector<json> extract_actions_robust(const std::string& text) const {
        return CodingAgentUtils::extract_actions_robust(text);
    }
    std::string run(const std::string& task, int max_turns = 6,
                    const std::vector<std::string>& expected_outputs = {},
                    const std::vector<json>& validation_checks = {}) {
        this->expected_outputs = expected_outputs;
        this->validation_checks = validation_checks;
        this->current_task = task;
        this->successful_plan_actions.clear();
        this->last_validation_feedback.clear();
        this->last_raw_error.clear();
        this->session_context.clear();
        this->current_turn_ = 0;
        this->current_max_turns_ = max_turns;
        this->llm_calls_at_start_ = llm.get_logical_call_count();
        critic.reset();
        critic.set_context(expected_outputs, validation_checks, task);
        perform_workspace_recon();
        const std::string task_hash = md5_hash(task);
        std::unordered_map<std::string, std::string> replay_artifacts;
        double replay_reward = -100.0;
        if (!expected_outputs.empty() &&
            semantic_memory.retrieve_replayable(task, expected_outputs, validation_checks,
                                                replay_artifacts, &replay_reward) &&
            replay_reward >= 2.0) {
            std::cout << "[Agent] Artifact-first replay from memory (reward="
                      << replay_reward << ")\n";
            for (const auto& [fp, content] : replay_artifacts) {
                std::string msg = tool_executor.file_manager.write_file(fp, content);
                std::cout << " [Artifact Replay] " << msg << "\n";
                if (std::find(critic.written_files().begin(), critic.written_files().end(), fp) ==
                    critic.written_files().end()) {
                    critic.written_files().push_back(fp);
                }
            }
            critic.invalidate_validation_cache_for("");
            if (critic.all_validations_pass())
                return succeed(task_hash,
                               "[Agent] Artifact replay satisfied all validations — terminating early. ");
            std::vector<std::string> art_fails = critic.collect_validation_failures();
            if (!art_fails.empty()) {
                last_validation_feedback = join(art_fails, "\n");
                session_context.push_back("[Artifact Replay Diagnostics]:\n" + last_validation_feedback);
            }
        }
        double cached_reward = -100.0;
        std::string cached = semantic_memory.retrieve_similar(task, 0.35, &cached_reward);
        if (!cached.empty() && cached_reward >= 2.0) {
            std::vector<json> plan = agent_planner.extract_replayable_plan(cached);
            if (plan.empty()) {
                std::cout << "[Agent] Cached entry has no replayable plan; falling back to LLM replan.\n";
            } else {
                std::cout << "[Agent] Replaying cached plan (" << plan.size()
                          << " actions, Expected Reward=" << cached_reward << ")\n";
                auto [score, missing] =
                    agent_planner.score_plan_against_validation(plan, expected_outputs, validation_checks);
                if (score < 0.4) {
                    std::cout << "[Agent] Cached plan scored too low (" << score
                              << "); falling back to LLM replan.\n";
                } else {
                    if (score < 0.95)
                        plan = agent_planner.repair_plan_suffix(plan, missing, task,
                                                                build_validation_requirements());
                    execute_plan(plan);
                    if (critic.all_validations_pass())
                        return succeed(task_hash,
                                       "[Agent] Cached plan satisfied all validations — terminating early. ");
                    std::vector<std::string> rep_fails = critic.collect_validation_failures();
                    if (!rep_fails.empty()) {
                        last_validation_feedback = join(rep_fails, "\n");
                        session_context.push_back("[Cached Plan Diagnostics - CEGIS Fix Required]:\n" +
                                                  last_validation_feedback);
                    }
                }
            }
        } else if (!cached.empty()) {
            std::cout << "[Agent] Cached plan has low historical reward (" << cached_reward
                      << "). Forcing LLM replan.\n";
        }
        const json tool_call_schema = CodingAgentUtils::build_tool_schema();
        if (!critic.all_validations_pass()) {
            std::cout << "[Agent] Generating validation-conditioned plan...\n";
            std::vector<json> plan =
                agent_planner.generate_validation_conditioned_plan(task, expected_outputs, validation_checks);
            auto [plan_score, missing_files] =
                agent_planner.score_plan_against_validation(plan, expected_outputs, validation_checks);
            if (plan_score < 0.95 || !missing_files.empty())
                plan = agent_planner.repair_plan_suffix(plan, missing_files, task,
                                                        build_validation_requirements());
            if (!plan.empty()) {
                auto sub_goals = agent_planner.decompose_plan(plan);
                if (!sub_goals.empty()) {
                    std::cout << "[Agent] Plan-derived decomposition into " << sub_goals.size()
                              << " sub-goals.\n";
                    session_context.push_back("Strategic Plan Breakdown: " + join(sub_goals, ", "));
                }
                execute_plan(plan);
                if (critic.all_validations_pass())
                    return succeed(task_hash,
                                   "[Agent] Initial plan execution satisfied all validations — terminating early. ");
            }
            std::vector<std::string> init_fails = critic.collect_validation_failures();
            critic.update_min_failures(static_cast<int>(init_fails.size()));
            if (!init_fails.empty()) {
                std::cout << "[Agent] Closed-loop CEGIS diagnostic repair for " << init_fails.size()
                          << " failure(s)...\n";
                const std::string heal_prompt =
                    "TASK: " + task + "\n" + build_validation_requirements() + "\n" +
                    "Execution validation failed with the following CEGIS counterexamples:\n" +
                    join(init_fails, "\n") + "\n" +
                    build_artifact_context() +
                    "Generate the exact write_file tool action that rewrites the failing file to satisfy all runtime validation outputs exactly. "
                    "Return exactly ONE JSON object with 'tool' and 'args'.";
                json heal_action = llm.structured_chat(
                    {{"system", CodingAgentUtils::system_prompt()}, {"user", heal_prompt}},
                    tool_call_schema, 0.0, 4096);
                if (heal_action.is_object() && heal_action.contains("tool")) {
                    dispatch_action(normalize_and_fix_action(heal_action), "Heal");
                    if (critic.all_validations_pass())
                        return succeed(task_hash, "[Agent] Closed-loop CEGIS validation healing succeeded! ");
                }
            }
        }
        int turn = 0;
        while (turn < max_turns && !aborted) {
            current_turn_ = turn + 1;
            if (critic.no_progress_turns() >= 2) {
                json recovery = critic.synthesize_recovery_action();
                if (!recovery.empty()) {
                    std::cout << "[Agent] No progress for " << critic.no_progress_turns()
                              << " turns — executing targeted active recovery action.\n";
                    dispatch_action(recovery, "Recovery");
                    if (critic.all_validations_pass())
                        return succeed(task_hash, "[Agent] Deterministic recovery satisfied validations. ");
                }
            }
            std::string user_msg = build_user_message(task, turn, max_turns);
            std::cout << "\n[Turn " << turn + 1 << "] Querying LLM (structuredCEGIS)...\n";
            std::vector<json> actions;
            json structured_action = llm.structured_chat(
                {{"system", CodingAgentUtils::system_prompt()}, {"user", user_msg}}, tool_call_schema,
                0.0, 4096);
            if (structured_action.is_object() && structured_action.contains("tool")) {
                std::cout << "[Turn " << turn + 1
                          << "] Structured action: " << structured_action.dump().substr(0, 300) << "\n";
                actions.push_back(structured_action);
            } else {
                std::cout << "[Turn " << turn + 1 << "] Structured empty/invalid, trying raw...\n";
                auto chat_res =
                    llm.chat({{"system", CodingAgentUtils::system_prompt()}, {"user", user_msg}}, 0.0,
                             4096);
                if (chat_res.ok) {
                    actions = extract_actions_robust(chat_res.content);
                }
                if (actions.empty()) {
                    std::cout << "[Turn " << turn + 1 << "] No action parsed.\n";
                    json recovery = critic.synthesize_recovery_action();
                    if (!recovery.empty()) {
                        std::cout << "[Agent] Substituting deterministic recovery for unparsed turn.\n";
                        actions.push_back(recovery);
                    }
                }
            }
            bool finish_called = false;
            for (size_t ai = 0; ai < actions.size(); ++ai) {
                json action = normalize_and_fix_action(actions[ai]);
                const std::string tool_name = action.value("tool", "");
                if (tool_name == "finish") {
                    finish_called = true;
                    break;
                }
                if (critic.should_skip_duplicate_action(action)) {
                    std::cout << "[Guard] Blocked duplicate action: " << tool_name << "\n";
                    critic.blocked_action_count()++;
                    session_context.push_back(
                        "System Warning: Blocked REDUNDANT call to '" + tool_name + "' on '" +
                        critic.get_action_target_file(action) +
                        "'. Nothing changed since the last identical attempt. You MUST pick a DIFFERENT action or structural algorithm variant.\n");
                    continue;
                }
                std::cout << "[Turn " << turn + 1 << "." << ai + 1 << "] Executing: " << tool_name << "\n";
                dispatch_action(action, "Turn " + std::to_string(turn + 1) + "." + std::to_string(ai + 1));
                if (critic.all_validations_pass())
                    return succeed(task_hash, "[Agent] Validation passed — terminating early. ");
            }
            if (finish_called) {
                if (critic.all_validations_pass())
                    return succeed(task_hash, "[Agent] Finish called and validations pass. ");
                last_validation_feedback = join(critic.collect_validation_failures(), "\n");
                session_context.push_back("[Validation CEGIS Diagnostic - Fix Required]:\n" +
                                          last_validation_feedback);
            } else {
                std::vector<std::string> fails = critic.collect_validation_failures();
                if (!fails.empty()) {
                    int current_fail_count = static_cast<int>(fails.size());
                    if (current_fail_count > critic.min_validation_failures()) {
                        std::cout << "[Backtracking] Validation failures increased ("
                                  << critic.min_validation_failures() << " -> " << current_fail_count
                                  << "). Rolling back to latest file backups if available.\n";
                        int restored = 0;
                        for (const auto& f : critic.written_files()) {
                            std::string msg = tool_executor.file_manager.restore_latest_backup(f);
                            if (!msg.empty()) {
                                std::cout << "  " << msg << "\n";
                                critic.invalidate_validation_cache_for(f);
                                ++restored;
                            }
                        }
                        if (restored == 0) {
                            std::cout << "  (no prior backups available for any written file)\n";
                        }
                    }
                    critic.update_min_failures(current_fail_count);
                    last_validation_feedback = join(fails, "\n");
                }
            }
            critic.update_progress();
            ++turn;
        }
        critic.set_llm_call_count(static_cast<int>(llm.get_logical_call_count() - llm_calls_at_start_));
        critic.compute_and_store_metrics(false, current_turn_, max_turns,
                                         current_task, successful_plan_actions);
        semantic_memory.report_outcome(task_hash, false);
        return "FAILED: Max turns reached without passing all validations. ";
    }
    void abort() {
        aborted = true;
        llm.abort();
    }
    void reset_abort() {
        aborted = false;
        llm.reset_abort();
    }
    SemanticMemoryManager semantic_memory;
    LLMBackendManager llm;
    ToolExecutor tool_executor;
    TaskDecomposer decomposer;
    DependencyPlanner planner_dep;
    AgentCritic critic;
    AgentPlanner agent_planner;
    std::vector<std::string> session_context;
    std::string last_validation_feedback, last_raw_error, current_task;
    std::vector<std::string> expected_outputs;
    std::vector<json> validation_checks, successful_plan_actions;
    std::atomic<bool> aborted{false};
    int current_turn_ = 0;
    int current_max_turns_ = 6;
    long long llm_calls_at_start_ = 0;

private:
    std::string succeed(const std::string& task_hash, const std::string& message) {
        std::cout << message << "\n";
        std::unordered_map<std::string, std::string> artifacts;
        for (const auto& fp : critic.written_files()) {
            if (!fs::exists(fp)) continue;
            std::ifstream in(fp, std::ios::binary);
            if (!in.is_open()) continue;
            std::ostringstream ss;
            ss << in.rdbuf();
            artifacts[fp] = ss.str();
        }
        if (!artifacts.empty()) {
            json plan_json = json::array();
            for (const auto& a : successful_plan_actions) plan_json.push_back(a);
            json sol = {{"plan", plan_json}};
            semantic_memory.store(current_task, sol.dump(), 0.0, artifacts);
        }
        critic.set_llm_call_count(static_cast<int>(llm.get_logical_call_count() - llm_calls_at_start_));
        critic.compute_and_store_metrics(true, current_turn_, current_max_turns_,
                                         current_task, successful_plan_actions);
        semantic_memory.report_outcome(task_hash, true);
        return "SUCCESS: All expected outputs and validation checks passed. ";
    }
    std::string dispatch_action(const json& action, const std::string& tag) {
        const std::string tool = action.value("tool", "");
        const std::string target_fp = critic.get_action_target_file(action);
        if (tool == "write_file" && !target_fp.empty()) {
            tool_executor.file_manager.backup_file(target_fp);
        }
        std::string observation = tool_executor.execute_action_dict(action);
        critic.record_executed_action(action);
        session_context.push_back(tag + ": " + tool + "\n→ " + observation.substr(0, 500));
        successful_plan_actions.push_back(action);
        if (tool == "write_file") {
            const std::string fp = action.value("args", json::object()).value("file_path", "");
            if (!fp.empty() &&
                std::find(critic.written_files().begin(), critic.written_files().end(), fp) ==
                    critic.written_files().end()) {
                critic.written_files().push_back(fp);
            }
            if (observation.find("File written successfully") != std::string::npos) {
                critic.invalidate_validation_cache_for(fp);
                auto_execute_after_write(fp);
            }
        }
        if (tool == "run_shell_command" || tool == "install_package") {
            critic.invalidate_validation_cache_for("");
        }
        if (CodingAgentUtils::is_critical_error(observation)) {
            last_raw_error = observation;
            std::vector<std::string> fixes;
            if (tool == "compile_cpp" || tool == "execute_file" || tool == "write_file")
                fixes.push_back("write_file");
            semantic_memory.store_failure(md5_hash(current_task),
                                          CodingAgentUtils::error_type_from_output(observation),
                                          observation, fixes, CodingAgentUtils::get_error_hint(observation));
        }
        return observation;
    }
    void auto_execute_after_write(const std::string& fp) {
        if (fp.empty())
            return;
        for (const auto& c : validation_checks) {
            if (c.value("type", "") != "execution")
                continue;
            if (c.value("file", "") != fp)
                continue;
            const std::string stdin_input = c.value("input", "");
            const std::string out = critic.execute_cached(fp, stdin_input);
            session_context.push_back("[Auto-Verify CEGIS] Executed " + fp +
                                      " immediately after write.\n→ " + out.substr(0, 500));
            const std::string expect = c.value("expect", "");
            if (!expect.empty()) {
                std::string lo = out, le = expect;
                std::transform(lo.begin(), lo.end(), lo.begin(), ::tolower);
                std::transform(le.begin(), le.end(), le.begin(), ::tolower);
                if (lo.find(le) == std::string::npos) {
                    session_context.push_back(
                        "[Auto-Verify CEGIS] MISSING REQUIRED OUTPUT: stdout did not contain '" +
                        expect + "'. Input stdin was '" + stdin_input + "'. Rewrite " + fp +
                        " to explicitly print that exact string at runtime.\n");
                }
            }
        }
    }
    std::string build_user_message(const std::string& task, int turn, int max_turns) {
        std::string msg = "TASK: " + task + "\n";
        msg += build_validation_requirements() + "\n";
        msg += build_file_exists_hint() + "\n";
        msg += build_progress_hint() + "\n";
        if (!last_validation_feedback.empty()) {
            msg += "CEGIS COUNTEREXAMPLE FEEDBACK:\n" + last_validation_feedback + "\n";
            msg += build_artifact_context() + "\n";
        }
        msg += "HISTORY:\n" + join_last(session_context, 8) + "\n";
        std::string rl_penalty = critic.build_action_penalty_hint();
        if (!rl_penalty.empty())
            msg += rl_penalty;
        msg += "Turn " + std::to_string(turn + 1) + "/" + std::to_string(max_turns) +
               ". Return exactly ONE JSON tool call.";
        return msg;
    }
    std::string build_artifact_context() const {
        std::string ctx;
        std::set<std::string> seen;
        for (const auto& c : validation_checks) {
            if (c.value("type", "") != "execution")
                continue;
            const std::string fp = c.value("file", "");
            if (fp.empty() || seen.count(fp) || !fs::exists(fp))
                continue;
            std::ifstream in(fp, std::ios::binary);
            if (!in.is_open())
                continue;
            std::ostringstream ss;
            ss << in.rdbuf();
            std::string content = ss.str();
            if (content.size() > 12000) {
                content = content.substr(0, 12000) + "\n...[TRUNCATED]...\n";
            }
            ctx += "CURRENT ARTIFACT '" + fp + "' (full content; patch this exact file, do not emit fragments):\n";
            ctx += content;
            if (!content.empty() && content.back() != '\n')
                ctx += "\n";
            ctx += "END ARTIFACT '" + fp + "'\n";
            seen.insert(fp);
        }
        return ctx;
    }
    std::string perform_workspace_recon() {
        std::string recon = "[WorkspaceRecon]\n";
        return recon;
    }
    json normalize_and_fix_action(const json& action) const {
        json a = action;
        if (!a.contains("args") || !a["args"].is_object())
            a["args"] = json::object();
        for (auto it = a.begin(); it != a.end(); ) {
            if (it.key() != "tool" && it.key() != "args") {
                a["args"][it.key()] = it.value();
                it = a.erase(it);
            } else ++it;
        }
        return a;
    }
    std::string build_file_exists_hint() const {
        std::string h;
        for (const auto& f : expected_outputs) {
            if (!fs::exists(f))
                h += "Missing expected file: " + f + "\n";
        }
        return h;
    }
    std::string build_progress_hint() const {
        return "Progress: " + std::to_string(critic.progress_ratio() * 100) + "%\n";
    }
    std::string build_validation_requirements() const {
        std::string req = "VALIDATION REQUIREMENTS (CEGIS):\n";
        for (const auto& c : validation_checks) {
            if (c.value("type", "") == "execution") {
                req += "- File " + c.value("file", "") +
                       " must execute successfully with input stdin '" + c.value("input", "") + "'";
                if (c.contains("expect"))
                    req += " and print expected substring '" + c.value("expect", "") + "'";
                req += "\n";
            }
        }
        return req;
    }
    void execute_plan(const std::vector<json>& raw_plan) {
        for (size_t i = 0; i < raw_plan.size(); ++i) {
            json a = normalize_and_fix_action(raw_plan[i]);
            const std::string tool = a.value("tool", "");
            if (tool == "finish")
                break;
            if (critic.should_skip_duplicate_action(a))
                continue;
            std::cout << " [Plan Step " << (i + 1) << "/" << raw_plan.size() << "] " << tool
                      << " (CEGIS-DAG)\n";
            dispatch_action(a, "PlanStep");
            if (critic.all_validations_pass()) {
                std::cout << "[Agent] CEGIS-DAG dynamic early-exit: all validations satisfied.\n";
                break;
            }
        }
    }
    static std::string join(const std::vector<std::string>& vec, const std::string& delim) {
        std::string result;
        for (size_t i = 0; i < vec.size(); ++i) {
            result += vec[i];
            if (i + 1 < vec.size())
                result += delim;
        }
        return result;
    }
    static std::string join_last(const std::vector<std::string>& vec, size_t n) {
        if (vec.empty())
            return "";
        size_t start = vec.size() > n ? vec.size() - n : 0;
        std::string result;
        for (size_t i = start; i < vec.size(); ++i) {
            result += vec[i];
            if (i + 1 < vec.size())
                result += "\n";
        }
        return result;
    }
};

static void run_single_task(CodingAgent& agent) {
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cout << "Enter task description: ";
    std::string task;
    std::getline(std::cin, task);
    if (task.empty())
        return;
    std::cout << "Expected output files (comma-separated, optional): ";
    std::string expected_line;
    std::getline(std::cin, expected_line);
    std::vector<std::string> expected_outputs;
    size_t pos = 0;
    while ((pos = expected_line.find(',')) != std::string::npos) {
        std::string f = expected_line.substr(0, pos);
        if (!f.empty())
            expected_outputs.push_back(f);
        expected_line.erase(0, pos + 1);
    }
    if (!expected_line.empty())
        expected_outputs.push_back(expected_line);
    std::vector<json> checks;
    if (!expected_outputs.empty()) {
        std::cout << "Add validation checks? (y/n): ";
        std::string add;
        std::getline(std::cin, add);
        if (add == "y" || add == "yes") {
            for (auto& fp : expected_outputs) {
                if (fp.ends_with(".py") || fp.ends_with(".cpp")) {
                    std::cout << "Expected output substring for " << fp << " (optional): ";
                    std::string expect;
                    std::getline(std::cin, expect);
                    std::cout << "Stdin input for " << fp << " (optional): ";
                    std::string stdin_str;
                    std::getline(std::cin, stdin_str);
                    json check = {{"type", "execution"}, {"file", fp}};
                    if (!expect.empty())
                        check["expect"] = expect;
                    if (!stdin_str.empty())
                        check["input"] = stdin_str;
                    checks.push_back(check);
                } else {
                    checks.push_back({{"type", "file_exists"}, {"path", fp}});
                }
            }
        }
    }
    std::cout << "Max turns (default 6): ";
    std::string mt;
    std::getline(std::cin, mt);
    int max_turns = 6;
    if (!mt.empty()) {
        try {
            size_t idx = 0;
            int v = std::stoi(mt, &idx);
            if (idx == mt.size() && v > 0 && v <= 1000) {
                max_turns = v;
            } else {
                std::cout << "Invalid max turns '" << mt
                          << "'; using default " << max_turns << ".\n";
            }
        } catch (const std::exception&) {
            std::cout << "Invalid max turns '" << mt
                      << "'; using default " << max_turns << ".\n";
        }
    }
    std::string result = agent.run(task, max_turns, expected_outputs, checks);
    std::cout << "\nResult: " + result << "\n";
}

static void run_test_cases(CodingAgent& agent,
                           const std::vector<TestCase>& cases,
                           const std::vector<std::string>& test_ids) {
    int total_tests = 0, passed_tests = 0;
    for (auto& tc : cases) {
        if (!test_ids.empty() &&
            std::find(test_ids.begin(), test_ids.end(), tc.id) == test_ids.end())
            continue;
        ++total_tests;
        std::cout << "\n--- " << tc.id << " " << tc.name << " ---\n";
        agent.reset_abort();
        auto start = std::chrono::steady_clock::now();
        std::string output;
        std::atomic<bool> done{false};
        std::exception_ptr exc = nullptr;
        std::thread t([&] {
            try {
                output = agent.run(tc.task_prompt, tc.max_turns, tc.expected_outputs,
                                   tc.validation_checks);
            } catch (...) { exc = std::current_exception(); }
            done = true;
        });
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(tc.timeout);
        while (!done && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        double dur = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (!done) {
            std::cout << "⏳ Timed out after " << tc.timeout << "s\n";
            agent.abort();
            t.join();
        } else {
            t.join();
            if (exc) {
                std::cout << "❌ Exception: ";
                try { std::rethrow_exception(exc); }
                catch (const std::exception& e) { std::cout << e.what(); }
                catch (...) { std::cout << "unknown"; }
                std::cout << "\n";
            } else if (output.find("SUCCESS") != std::string::npos) {
                std::cout << "✅ Passed in " << dur << "s\n";
                ++passed_tests;
            } else {
                std::cout << "❌ Failed in " << dur << "s\n";
            }
        }
    }
    if (total_tests > 0) {
        double pass_rate = 100.0 * passed_tests / total_tests;
        std::cout << "\n================ TEST SUMMARY ====\n";
        std::cout << "Passed: " << passed_tests << "/" << total_tests << "\n";
        std::cout << "Pass rate: " << pass_rate << "%\n";
    } else {
        std::cout << "\nNo matching tests were run.\n";
    }
}

int main() {
    curl_global_init(CURL_GLOBAL_ALL);
    srand(static_cast<unsigned>(time(nullptr)));
    try {
        CodingAgent agent;
        TestSuite suite;
        while (true) {
            std::cout << "\n==================================================\n";
            std::cout << "1. Single task\n";
            std::cout << "2. Run standard tests (TestSuite)\n";
            std::cout << "3. Stats\n";
            std::cout << "4. Propose self-improvement\n";
            std::cout << "5. Quit\n";
            std::cout << "Choice: ";
            std::string choice;
            std::getline(std::cin, choice);
            if (choice == "5" || choice == "quit" || choice == "exit") {
                break;
            } else if (choice == "1") {
                run_single_task(agent);
            } else if (choice == "2") {
                std::cout << "Test IDs to run (comma-separated, or 'all'): ";
                std::string ids;
                std::getline(std::cin, ids);
                std::vector<std::string> test_ids;
                if (!ids.empty() && ids != "all") {
                    size_t pos = 0;
                    while ((pos = ids.find(',')) != std::string::npos) {
                        test_ids.push_back(ids.substr(0, pos));
                        ids.erase(0, pos + 1);
                    }
                    if (!ids.empty())
                        test_ids.push_back(ids);
                }
                if (choice == "2")
                    run_test_cases(agent, suite.get_cases(), test_ids);
            } else if (choice == "3") {
                agent.semantic_memory.print_stats();
            } else if (choice == "4") {
                std::cout << "\n[Self-Modification] Analyzing agent source code...\n";
                std::vector<std::string> own_files = {
                    "Advanced_Frontier_Coding_A.cpp"
                };
                SelfModificationEngine engine(agent.llm);
                json proposal = engine.proposeImprovement(own_files);
                std::cout << "\n=== PROPOSED IMPROVED PROPOSAL ===\n";
                std::cout << proposal.dump(2) << "\n";
                std::cout << "======================\n";
            } else {
                std::cout << "Invalid choice.\n";
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Initialization error: " << e.what() << "\n";
        std::cerr << "Make sure at least one API key is set:\n";
        std::cerr << "  GROQ_API_KEY, OPENROUTER_API_KEY, or GOOGLE_API_KEY\n";
        return 1;
    }
    curl_global_cleanup();
    return 0;
}
