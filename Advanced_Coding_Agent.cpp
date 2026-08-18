// You are an advanced AI coding agent specializing in algorithmic discovery and 
// optimization, powered by cutting-edge LLMs. Your mission is to take an initial 
// C++ algorithm, analyze its purpose, understand simulated evaluation results, 
// and propose a single, significant improvement. Print the full corrected class 
// or definition that are modified.
// Requires: libcurl, nlohmann/json.hpp
// Compile: g++ -std=c++20 -O2 -pthread Advanced_Coding_Agent_01.cpp -lcurl -o AdvancedCodingAgent_01
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <thread>
#include <mutex>
#include <functional>
#include <regex>
#include <filesystem>
#include <cstdlib>
#include <ctime>
#include <atomic>
#include <queue>
#include <random>
#include <memory>
#include <stdexcept>
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <sys/wait.h>
#include <unistd.h>

using json = nlohmann::json;
namespace fs = std::filesystem;

static double get_time() {
    return std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

static std::string md5_hash(const std::string &s) {
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : s) h = (h ^ c) * 1099511628211ULL;
    std::ostringstream oss;
    oss << std::hex << (h & 0xFFFFFFFFULL);
    return oss.str();
}

static const std::unordered_set<std::string> STOP_WORDS = {
    "the","and","is","in","to","of","a","for","on","with","as","by","at","an","be",
    "this","that","it","not","or","but","are","from","has","had","have","will","would",
    "could","should","may","can","do","does","did","was","were","been","being","am","i",
    "you","he","she","we","they","me","him","her","us","them","my","your","his","its",
    "our","their","what","when","where","why","how","all","any","both","each","few","more",
    "most","other","some","such","no","nor","only","own","same","so","than","through",
    "too","under","until","up","very","with","would"
};

static std::vector<std::string> tokenize(const std::string &text) {
    std::string lower = text;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    std::regex re(R"(\b[a-z]{3,}\b)");
    std::sregex_iterator it(lower.begin(), lower.end(), re), end;
    std::vector<std::string> tokens;
    for (; it != end; ++it) {
        std::string w = it->str();
        if (!STOP_WORDS.count(w)) tokens.push_back(w);
    }
    return tokens;
}

static std::vector<std::pair<std::string, double>> compute_tfidf_vector(
    const std::string &text,
    const std::unordered_map<std::string, double> &idf,
    int total_docs) {
    auto tokens = tokenize(text);
    std::unordered_map<std::string, int> freq;
    for (auto &t : tokens) freq[t]++;
    if (freq.empty()) return {};
    int max_freq = 0;
    for (auto &[_, c] : freq) max_freq = std::max(max_freq, c);
    std::vector<std::pair<std::string, double>> vec;
    for (auto &[term, count] : freq) {
        double tf = static_cast<double>(count) / max_freq;
        double idf_val = 1.0;
        if (auto it = idf.find(term); it != idf.end()) idf_val = it->second;
        else if (total_docs > 0) idf_val = std::log(1.0 + total_docs + 1);
        vec.push_back({term, tf * idf_val});
    }
    std::sort(vec.begin(), vec.end());
    return vec;
}

static std::string error_type_from_output(const std::string &output) {
    std::string low = output;
    std::transform(low.begin(), low.end(), low.begin(), ::tolower);
    if (low.find("no such file") != std::string::npos) return "missing_file";
    if (low.find("name or service not known") != std::string::npos ||
        low.find("name resolution") != std::string::npos) return "name_resolution";
    if (low.find("segmentation fault") != std::string::npos ||
        low.find("return code: 139") != std::string::npos ||
        low.find("return code: -11") != std::string::npos) return "segmentation_fault";
    if (low.find("return code: 1") != std::string::npos) return "execution_error";
    if (low.find("compilation") != std::string::npos ||
        low.find("g++") != std::string::npos) return "compilation_error";
    return "unknown";
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

static size_t WriteCallback(void *contents, size_t size, size_t nmemb, std::string *out) {
    size_t total = size * nmemb;
    out->append(static_cast<char*>(contents), total);
    return total;
}

static bool http_post_json(const std::string &url,
                           const std::vector<std::string> &header_lines,
                           const json &payload,
                           int timeout_sec,
                           std::string &response,
                           std::string *error_out = nullptr) {
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
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout_sec);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
    curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, curl_errbuf);
    CURLcode res = curl_easy_perform(curl);
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    std::string error;
    if (res != CURLE_OK) {
        error = std::string(curl_errbuf[0] ? curl_errbuf : curl_easy_strerror(res));
        error = "curl error: " + error;
    } else if (http_code != 200) {
        error = "HTTP " + std::to_string(http_code) + ": " + response.substr(0, 300);
    }
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    if (!error.empty()) {
        if (error_out) *error_out = error;
        else std::cerr << "[HTTP] " << url << " -> " << error << "\n";
        return false;
    }
    return true;
}

struct ChatResult {
    std::string content;
    bool ok = false;
    std::string error;
};

class LLMClientBase {
public:
    explicit LLMClientBase(std::string api_key) : api_key(std::move(api_key)) {}
    virtual ~LLMClientBase() = default;
    virtual ChatResult fetch(const std::string &model,
                             const std::vector<std::pair<std::string, std::string>> &messages,
                             double temperature,
                             int max_tokens) = 0;
    ChatResult chat(const std::string &model,
                    const std::vector<std::pair<std::string, std::string>> &messages,
                    double temperature = 0.0,
                    int max_tokens = 2048,
                    int max_attempts = 3) {
        for (int attempt = 0; attempt < max_attempts; ++attempt) {
            try {
                ChatResult r = fetch(model, messages, temperature, max_tokens);
                if (r.ok) return r;
                if (attempt == max_attempts - 1) return r;
            } catch (const std::exception &e) {
                if (attempt == max_attempts - 1) {
                    return {"", false, e.what()};
                }
            }
            double delay = std::min(60.0, std::pow(2.0, attempt)) * (0.5 + (rand() % 1000) / 1000.0);
            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<long long>(delay * 1000)));
        }
        return {"", false, "Max retries exceeded"};
    }
protected:
    std::string api_key;
};

class GroqClient : public LLMClientBase {
public:
    explicit GroqClient(const std::string &key) : LLMClientBase(key) {}
    ChatResult fetch(const std::string &model,
                     const std::vector<std::pair<std::string, std::string>> &messages,
                     double temperature,
                     int max_tokens) override {
        json msgs = json::array();
        for (auto &[role, content] : messages) {
            msgs.push_back({{"role", role}, {"content", content}});
        }
        json payload = {
            {"model", model},
            {"temperature", temperature},
            {"max_tokens", max_tokens},
            {"messages", msgs}
        };
        std::string resp;
        if (!http_post_json("https://api.groq.com/openai/v1/chat/completions",
                            {"Authorization: Bearer " + api_key,
                             "Content-Type: application/json"},
                            payload, 90, resp)) {
            return {"", false, "HTTP error or network failure"};
        }
        try {
            json data = json::parse(resp);
            return {data["choices"][0]["message"]["content"], true, ""};
        } catch (...) {
            return {"", false, "Unexpected response: " + resp};
        }
    }
};

class OpenRouterClient : public LLMClientBase {
public:
    OpenRouterClient(const std::string &key, int rpm = 30)
        : LLMClientBase(key), limiter(rpm) {}
    ChatResult fetch(const std::string &model,
                     const std::vector<std::pair<std::string, std::string>> &messages,
                     double temperature,
                     int max_tokens) override {
        limiter.acquire();
        json msgs = json::array();
        for (auto &[role, content] : messages) {
            msgs.push_back({{"role", role}, {"content", content}});
        }
        json payload = {
            {"model", model},
            {"temperature", temperature},
            {"max_tokens", max_tokens},
            {"messages", msgs}
        };
        std::string resp;
        if (!http_post_json("https://openrouter.ai/api/v1/chat/completions",
                            {"Authorization: Bearer " + api_key,
                             "Content-Type: application/json",
                             "HTTP-Referer: https://localhost",
                             "X-Title: CodingAgent"},
                            payload, 90, resp)) {
            return {"", false, "HTTP error or network failure"};
        }
        try {
            json data = json::parse(resp);
            return {data["choices"][0]["message"]["content"], true, ""};
        } catch (...) {
            return {"", false, "Unexpected response: " + resp};
        }
    }
private:
    RateLimiter limiter;
};

class GoogleClient : public LLMClientBase {
public:
    explicit GoogleClient(const std::string &key) : LLMClientBase(key) {}
	ChatResult fetch(const std::string &model,
	                 const std::vector<std::pair<std::string, std::string>> &messages,
	                 double temperature,
	                 int max_tokens) override {
	    std::string url =
	        "https://generativelanguage.googleapis.com/v1beta/models/" +
	        model + ":generateContent?key=" + api_key;
	    json contents = json::array();
	    json systemInstruction;
	    for (auto &[role, text] : messages) {
	        if (role == "system") {
	            json part = json::object();
	            part["text"] = text;
	
	            systemInstruction = json::object();
	            systemInstruction["parts"] = json::array({part});
	        } else {
	            std::string mapped = (role == "user") ? "user" : "model";
	
	            if (!contents.empty() && contents.back()["role"] == mapped) {
	                contents.back()["parts"][0]["text"] =
	                    contents.back()["parts"][0]["text"].get<std::string>() +
	                    "\n\n" + text;
	            } else {
	                json part = json::object();
	                part["text"] = text;
	
	                json item = json::object();
	                item["role"] = mapped;
	                item["parts"] = json::array({part});
	
	                contents.push_back(item);
	            }
	        }
	    }
	    if (!contents.empty() && contents[0]["role"] == "model") {
	        json part = json::object();
	        part["text"] = "Continue your work.";
	
	        json item = json::object();
	        item["role"] = "user";
	        item["parts"] = json::array({part});
	
	        contents.insert(contents.begin(), item);
	    }
	    json payload = {
	        {"contents", contents},
	        {"generationConfig", {
	            {"temperature", temperature},
	            {"maxOutputTokens", max_tokens}
	        }}
	    };
	    if (!systemInstruction.is_null()) {
	        payload["systemInstruction"] = systemInstruction;
	    }
	    std::string resp;
	    if (!http_post_json(url,
	                        {"Content-Type: application/json"},
	                        payload, 90, resp)) {
	        return {"", false, "HTTP error or network failure"};
	    }
	    try {
	        json data = json::parse(resp);
	        if (data.contains("candidates") &&
	            data["candidates"].is_array() &&
	            !data["candidates"].empty()) {
	            return {
	                data["candidates"][0]["content"]["parts"][0]["text"],
	                true,
	                ""
	            };
	        }
	        return {"", false, "No candidates: " + resp};
	    } catch (...) {
	        return {"", false, "Unexpected response: " + resp};
	    }
	}    
};

class LLMBackendManager {
public:
    struct Config {
        std::map<std::string, std::string> models;
        std::vector<std::string> priority;
        int global_rpm = 30;
    };
    LLMBackendManager(const Config &cfg) : config(cfg), limiter(cfg.global_rpm) {
        const char *groq = getenv("GROQ_API_KEY");
        const char *openrouter = getenv("OPENROUTER_API_KEY");
        const char *google = getenv("GOOGLE_API_KEY");
        if (groq) clients["groq"] = std::make_shared<GroqClient>(groq);
        if (openrouter) clients["openrouter"] = std::make_shared<OpenRouterClient>(openrouter);
        if (google) clients["google"] = std::make_shared<GoogleClient>(google);
        for (auto &b : config.priority) {
            if (clients.count(b)) { current_backend = b; break; }
        }
        if (current_backend.empty()) throw std::runtime_error("No LLM backend available");
    }
    void abort() { aborted = true; }
    void reset_abort() { aborted = false; }
    void reset_cooldowns() { cooldown_until.clear(); }
    ChatResult chat(const std::vector<std::pair<std::string, std::string>> &messages,
                    double temperature = 0.0,
                    int max_tokens = 2048) {
        if (aborted) return {"", false, "Aborted by agent."};
        limiter.acquire();
        if (aborted) return {"", false, "Aborted by agent."};
        std::vector<std::string> order = {current_backend};
        for (auto &b : config.priority) if (b != current_backend) order.push_back(b);
        for (int retry = 0; retry < 3; ++retry) {
            bool tried_any = false;
            for (auto &backend : order) {
                if (aborted) return {"", false, "Aborted by agent."};
                if (is_on_cooldown(backend)) continue;
                tried_any = true;
                auto it = clients.find(backend);
                if (it == clients.end()) continue;
                std::string model = config.models.count(backend) ? config.models.at(backend) : "unknown";                
                ChatResult r = it->second->chat(model, messages, temperature, max_tokens);
				if (r.ok) {
				    clear_all_cooldowns();
				    current_backend = backend;
				    return r;
				}
				std::cerr << "[BACKEND] " << backend << " failed: " << r.error << "\n";
				set_cooldown(backend, r.error);  
            }
            if (!tried_any) {
                double wait = min_cooldown_remaining();
                if (wait <= 0) continue;
                std::cout << "[BACKEND] All backends on cooldown, waiting "
                          << wait << "s\n";
                std::this_thread::sleep_for(std::chrono::milliseconds(
                    static_cast<long long>(wait * 1000)));
            } else {
                double sleep_time = std::min(15.0, 3.0 * std::pow(2.0, retry));
                std::this_thread::sleep_for(std::chrono::milliseconds(
                    static_cast<long long>(sleep_time * 1000)));
            }
        }
        return {"", false, "All backends exhausted."};
    }
    json structured_chat(const std::vector<std::pair<std::string, std::string>> &messages,
                         const json &schema,
                         double temperature = 0.0,
                         int max_tokens = 2048) {
        std::string system =
            "You are a precise JSON-only assistant. Output ONLY valid JSON matching the schema below.\n"
            "Schema:\n" + schema.dump(2) + "\nOutput the raw JSON immediately.";
        std::vector<std::pair<std::string, std::string>> full = {{"system", system}};
        full.insert(full.end(), messages.begin(), messages.end());
        ChatResult r = chat(full, temperature, max_tokens);
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
            std::string sanitized = smart_sanitize_for_structured(content);
            try { return json::parse(sanitized); } catch (...) {}
        }
        return json::object();
    }
    std::string get_current_backend() const { return current_backend; }

private:
    Config config;
    std::map<std::string, std::shared_ptr<LLMClientBase>> clients;
    std::string current_backend;
    std::map<std::string, double> cooldown_until;
    RateLimiter limiter;
    std::atomic<bool> aborted{false};
    bool is_on_cooldown(const std::string &backend) {
        auto it = cooldown_until.find(backend);
        if (it == cooldown_until.end()) return false;
        if (get_time() >= it->second) { cooldown_until.erase(it); return false; }
        return true;
    }
    void set_cooldown(const std::string &backend, const std::string &err) {
        double now = get_time();
        std::string low = err;
        std::transform(low.begin(), low.end(), low.begin(), ::tolower);
        double wait = 2.0;
        if (low.find("quota") != std::string::npos) wait = 30.0;
        else if (low.find("429") != std::string::npos) wait = 10.0;
        cooldown_until[backend] = now + wait;
    }
    void clear_all_cooldowns() { cooldown_until.clear(); }
    double min_cooldown_remaining() {
        double now = get_time();
        double min_rem = 0.0;
        for (auto &[_, exp] : cooldown_until) {
            if (exp > now) {
                if (min_rem == 0.0 || exp - now < min_rem) min_rem = exp - now;
            }
        }
        return min_rem;
    }
    static std::string extract_json_with_bracket_matching(const std::string &text) {
        for (char start_char : {'{', '['}) {
            char end_char = (start_char == '{') ? '}' : ']';
            size_t idx = text.find(start_char);
            if (idx == std::string::npos) continue;
            int stack = 1;
            bool in_str = false;
            bool escape = false;
            size_t i = idx + 1;
            for (; i < text.size() && stack > 0; ++i) {
                char ch = text[i];
                if (escape) { escape = false; }
                else if (ch == '\\') { escape = true; }
                else if (ch == '"') { in_str = !in_str; }
                else if (!in_str) {
                    if (ch == start_char) stack++;
                    else if (ch == end_char) stack--;
                }
            }
            if (stack == 0) return text.substr(idx, i - idx);
        }
        return "";
    }
    static std::string smart_sanitize_for_structured(const std::string &text) {
        std::string result;
        bool in_string = false;
        bool escape_next = false;
        for (char ch : text) {
            if (escape_next) { result += ch; escape_next = false; continue; }
            if (ch == '\\') { result += ch; escape_next = true; continue; }
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
        std::regex trailing_comma(R"(,(\s*[}\]]))");
        result = std::regex_replace(result, trailing_comma, "$1");
        return result;
    }
};

class CodeExecutor {
public:
    int max_execution_timeout = 30;
    int max_install_timeout = 60;
    std::string last_compiled_exe;
    std::string run_shell_command(const std::string &command, int timeout = 30) {
        std::string full = "timeout " + std::to_string(timeout) + "s " + command +
                           " 2>&1; echo RC:$?";
        FILE *pipe = popen(full.c_str(), "r");
        if (!pipe) return "Error: cannot execute shell command";
        std::string output;
        char buffer[256];
        while (fgets(buffer, sizeof(buffer), pipe)) output += buffer;
        pclose(pipe);
        return "Shell output:\n" + output;
    }
    std::string compile_cpp(const std::string &source_file) {
        std::string out = "/tmp/agent_" + md5_hash(source_file) + ".out";
        std::string flags = "-std=c++17 -O2";
        std::ifstream f(source_file);
        std::string content((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());
        if (content.find("std::thread") != std::string::npos) flags += " -pthread";
        if (content.find("<curl/curl.h>") != std::string::npos) flags += " -lcurl";
        if (content.find("<openssl/") != std::string::npos) flags += " -lcrypto -lssl";
        if (content.find("<nlohmann/json.hpp>") != std::string::npos)
            flags += " -I/usr/include/nlohmann";
        std::string cmd = "g++ " + flags + " \"" + source_file + "\" -o \"" + out +
                          "\" 2>&1; echo RC:$?";
        std::string output = run_shell_command(cmd, 15);
        if (output.find("RC:0") == std::string::npos) {
            return "COMPILATION_FAILED: " + output;
        }
        last_compiled_exe = out;
        return out;
    }
    std::string execute_file(const std::string &file_path,
                             const std::string &stdin_input = "") {
        fs::path path(file_path);
        std::string ext = path.extension().string();
        std::string cmd;
        std::string exe_to_clean;
        if (ext.empty() && !last_compiled_exe.empty() &&
            fs::exists(last_compiled_exe)) {
            cmd = "\"" + last_compiled_exe + "\"";
        } else if (ext == ".py") {
            cmd = "python3 \"" + file_path + "\"";
        } else if (ext == ".cpp" || ext == ".cc" || ext == ".cxx") {
            std::string exe = compile_cpp(file_path);
            if (exe.find("COMPILATION_FAILED:") != std::string::npos) return exe;
            cmd = "\"" + exe + "\"";
            exe_to_clean = exe;
        } else if (fs::exists(path) && access(path.c_str(), X_OK) == 0) {
            cmd = "\"" + path.string() + "\"";
        } else {
            return "Unsupported file type: " + ext;
        }
        std::string stdin_file;
        if (!stdin_input.empty()) {
            stdin_file = "/tmp/agent_stdin_" + std::to_string(rand()) + ".txt";
            std::ofstream f(stdin_file);
            f << stdin_input;
            f.close();
        }
        std::string full;
        if (!stdin_file.empty()) {
            full = "timeout " + std::to_string(max_execution_timeout) + "s " + cmd +
                   " < \"" + stdin_file + "\" 2>&1; echo RC:$?";
        } else {
            full = "timeout " + std::to_string(max_execution_timeout) + "s " + cmd +
                   " 2>&1; echo RC:$?";
        }
        FILE *pipe = popen(full.c_str(), "r");
        std::string output;
        char buffer[512];
        while (fgets(buffer, sizeof(buffer), pipe)) output += buffer;
        pclose(pipe);
        if (!stdin_file.empty()) fs::remove(stdin_file);
        if (!exe_to_clean.empty() && fs::exists(exe_to_clean)) fs::remove(exe_to_clean);
        int rc = -1;
        size_t pos = output.rfind("RC:");
        if (pos != std::string::npos) {
            std::string rc_str = output.substr(pos + 3);
            rc = std::atoi(rc_str.c_str());
            output = output.substr(0, pos);
        }
        std::ostringstream oss;
        oss << "Return code: " << rc << "\nSTDOUT:\n" << output;
        return oss.str();
    }
    std::string execute_code(const std::string &code,
                             const std::string &language = "auto") {
        std::string lang = language;
        if (lang == "auto") {
            lang = (code.find("#include") != std::string::npos ||
                    code.find("int main(") != std::string::npos) ? "cpp" : "python";
        }
        std::string tmp = "/tmp/agent_code_" + std::to_string(rand()) +
                          ((lang == "cpp") ? ".cpp" : ".py");
        std::ofstream f(tmp);
        f << code;
        f.close();
        std::string result = execute_file(tmp);
        fs::remove(tmp);
        return result;
    }
    std::string validate_syntax(const std::string &code,
                                const std::string &language = "auto") {
        std::string lang = language;
        if (lang == "auto") {
            lang = (code.find("#include") != std::string::npos ||
                    code.find("int main(") != std::string::npos) ? "cpp" : "python";
        }
        if (lang == "cpp") {
            std::string tmp = "/tmp/agent_syntax_" + std::to_string(rand()) + ".cpp";
            std::ofstream f(tmp);
            f << code;
            f.close();
            std::string cmd = "g++ -std=c++20 -fsyntax-only \"" + tmp +
                              "\" 2>&1; echo RC:$?";
            std::string out = run_shell_command(cmd, 10);
            fs::remove(tmp);
            if (out.find("RC:0") != std::string::npos)
                return "Syntax validation PASSED.";
            return "Syntax validation FAILED: " + out;
        }
        std::string tmp = "/tmp/agent_syntax_" + std::to_string(rand()) + ".py";
        std::ofstream f(tmp);
        f << code;
        f.close();
        std::string cmd = "python3 -m py_compile \"" + tmp + "\" 2>&1; echo RC:$?";
        std::string out = run_shell_command(cmd, 10);
        fs::remove(tmp);
        if (out.find("RC:0") != std::string::npos)
            return "Syntax validation PASSED.";
        return "Syntax validation FAILED: " + out;
    }
};

struct SemanticMemoryEntry {
    std::string task_hash;
    std::string task_text;
    std::string solution_code;
    std::vector<std::pair<std::string, double>> word_vector;
    double norm = 0.0;
    double success_timestamp = 0.0;
    int api_calls_saved = 0;
    int access_count = 0;
    double last_access_time = 0.0;
    std::unordered_set<std::string> term_set;
    int term_count = 0;
    bool is_plan = false;
    int success_count = 0;
    int failure_count = 0;
};

class SemanticMemoryManager {
public:
    explicit SemanticMemoryManager(const std::string &db_path = "./agent_memory_db")
        : db_path(db_path) {
        fs::create_directories(db_path);
        load_all();
    }
    std::string retrieve_similar(const std::string &query,
                                 double threshold = 0.72) {
        std::string task_hash = md5_hash(query);
        if (auto it = exact.find(task_hash); it != exact.end()) {
            auto entry = it->second;
            double now = get_time();
            entry->access_count++;
            entry->last_access_time = now;
            entry->api_calls_saved++;
            save_all();
            stats.exact_hits++;
            stats.retrieval_count++;
            return entry->solution_code;
        }
        if (memory.empty()) {
            stats.misses++;
            return "";
        }
        auto query_vec = compute_tfidf_vector(query, idf, total_docs);
        if (query_vec.empty()) {
            stats.misses++;
            return "";
        }
        double query_norm = 0.0;
        for (auto &[_, w] : query_vec) query_norm += w * w;
        std::unordered_set<std::string> query_terms;
        for (auto &[t, _] : query_vec) query_terms.insert(t);
        double now = get_time();
        std::unordered_map<std::string, double> candidate_dot;
        std::unordered_map<std::string, std::shared_ptr<SemanticMemoryEntry>> candidate_map;
        for (auto &[term, q_weight] : query_vec) {
            if (auto it = inverted.find(term); it != inverted.end()) {
                for (auto &[entry, doc_weight] : it->second) {
                    candidate_dot[entry->task_hash] += q_weight * doc_weight;
                    candidate_map[entry->task_hash] = entry;
                }
            }
        }
        std::vector<std::tuple<std::shared_ptr<SemanticMemoryEntry>, double, double>> candidates;
        for (auto &[h, dot] : candidate_dot) {
            auto entry = candidate_map[h];
            double cos = 0.0;
            if (query_norm > 0 && entry->norm > 0)
                cos = dot / (std::sqrt(query_norm) * std::sqrt(entry->norm));
            double jac = jaccard(query_terms, entry->term_set);
            candidates.push_back({entry, cos, jac});
        }
        if (candidates.empty()) {
            for (auto &entry : memory)
                candidates.push_back({entry, 0.0, jaccard(query_terms, entry->term_set)});
        }
        std::shared_ptr<SemanticMemoryEntry> best_entry;
        double best_score = 0.0, best_cos = 0.0, best_jac = 0.0;
        for (auto &[entry, cos, jac] : candidates) {
            double semantic = hybrid(cos, jac);
            double age_days = (now - entry->success_timestamp) / 86400.0;
            double freshness = 1.0 - 0.1 * std::min(1.0, std::max(0.0, (age_days - 1.0) / 6.0));
            double reliability = memory_reliability(entry);
            double outcome_adjusted = 0.75 * semantic + 0.25 * reliability;
            double score = outcome_adjusted * freshness;
            if (score > best_score) {
                best_score = score;
                best_entry = entry;
                best_cos = cos;
                best_jac = jac;
            }
        }
        double eff_threshold = threshold;
        if (auto it = dynamic_thresholds.find(task_hash); it != dynamic_thresholds.end())
            eff_threshold = it->second;
        if (best_entry && best_score >= eff_threshold) {
            stats.semantic_hits++;
            stats.retrieval_count++;
            best_entry->access_count++;
            best_entry->last_access_time = now;
            best_entry->api_calls_saved++;
            last_retrieval[task_hash] = {best_cos, best_jac, best_entry->task_hash};
            save_all();
            return best_entry->solution_code;
        }
        stats.misses++;
        return "";
    }
    void store(const std::string &task, const std::string &solution) {
        std::string task_hash = md5_hash(task);
        bool is_plan = false;
        try {
            json parsed = json::parse(solution);
            if (parsed.is_array()) {
                is_plan = true;
                for (auto &a : parsed) if (!a.contains("tool")) { is_plan = false; break; }
            }
        } catch (...) {}
        std::shared_ptr<SemanticMemoryEntry> entry;
        if (auto it = exact.find(task_hash); it != exact.end()) {
            entry = it->second;
            entry->solution_code = solution;
        } else {
            entry = std::make_shared<SemanticMemoryEntry>();
            entry->task_hash = task_hash;
            entry->task_text = task;
            entry->solution_code = solution;
            memory.insert(memory.begin(), entry);
            exact[task_hash] = entry;
        }
        entry->success_timestamp = get_time();
        entry->is_plan = is_plan;
        entry->access_count++;
        if (static_cast<int>(memory.size()) > max_entries) {
            auto to_evict = std::min_element(
                memory.begin(), memory.end(),
                [&](const auto &a, const auto &b) {
                    double sa = 0.65 * std::min(a->access_count, 10) / 10.0 +
                                0.35 * memory_reliability(a);
                    double sb = 0.65 * std::min(b->access_count, 10) / 10.0 +
                                0.35 * memory_reliability(b);
                    return sa < sb;
                });
            if (to_evict != memory.end()) {
                exact.erase((*to_evict)->task_hash);
                memory.erase(to_evict);
            }
        }
        update_idf();
        save_all();
    }
    void store_failure(const std::string &task_hash,
                       const std::string &error_type,
                       const std::string &error_output,
                       const std::vector<json> &fix_actions,
                       const std::string &hint) {
        std::string sig = error_output.substr(0, 200);
        std::replace(sig.begin(), sig.end(), '\n', ' ');
        failure_patterns.push_back({
            task_hash, error_type, sig, fix_actions, hint, 1, get_time()
        });
        save_failures();
    }
    std::string get_hint_for_error(const std::string &error_output) {
        std::string etype = error_type_from_output(error_output);
        std::string sig = error_output.substr(0, 200);
        std::replace(sig.begin(), sig.end(), '\n', ' ');
        for (auto &p : failure_patterns) {
            if (p.error_type == etype &&
                (p.error_signature.find(sig) != std::string::npos ||
                 sig.find(p.error_signature) != std::string::npos)) {
                p.occurrence_count++;
                p.last_seen = get_time();
                save_failures();
                return p.hint;
            }
        }
        return "";
    }
    void report_outcome(const std::string &task_hash, bool success) {
        auto it = last_retrieval.find(task_hash);
        if (it != last_retrieval.end()) {
            auto [cos, jac, entry_hash] = it->second;
            if (auto eit = exact.find(entry_hash); eit != exact.end()) {
                auto entry = eit->second;
                if (success) entry->success_count++;
                else entry->failure_count++;
                double pred = (weights[0] * cos + weights[1] * jac) /
                              (weights[0] + weights[1] + 1e-12);
                double error = pred - (success ? 1.0 : 0.0);
                weights[0] -= learning_rate * error * cos;
                weights[1] -= learning_rate * error * jac;
                for (auto &w : weights) w = std::max(-5.0, std::min(5.0, w));
                save_all();
            }
            last_retrieval.erase(it);
        }
        double thresh = 0.72;
        if (success) thresh = std::max(0.50, thresh * 0.95);
        else {
            consec_fail[task_hash]++;
            if (consec_fail[task_hash] >= 2) {
                thresh = std::min(0.90, thresh * 1.05);
                consec_fail[task_hash] = 0;
            }
        }
        dynamic_thresholds[task_hash] = thresh;
    }
    void print_stats() const {
        int saved = 0;
        for (auto &e : memory) saved += e->api_calls_saved;
        std::cout << "Memory entries: " << memory.size() << "/" << max_entries << "\n";
        std::cout << "Hits exact=" << stats.exact_hits
                  << " semantic=" << stats.semantic_hits
                  << " misses=" << stats.misses << "\n";
        std::cout << "API calls saved: " << saved << "\n";
    }

private:
    struct FailurePattern {
        std::string task_hash;
        std::string error_type;
        std::string error_signature;
        std::vector<json> fix_actions;
        std::string hint;
        int occurrence_count = 1;
        double last_seen = 0.0;
    };
    struct Stats {
        int exact_hits = 0;
        int semantic_hits = 0;
        int misses = 0;
        int retrieval_count = 0;
    };
    std::vector<std::shared_ptr<SemanticMemoryEntry>> memory;
    std::unordered_map<std::string, std::shared_ptr<SemanticMemoryEntry>> exact;
    std::unordered_map<std::string, std::vector<std::pair<std::shared_ptr<SemanticMemoryEntry>, double>>> inverted;
    std::unordered_map<std::string, double> idf;
    int total_docs = 0;
    Stats stats;
    std::unordered_map<std::string, double> dynamic_thresholds;
    int max_entries = 50;
    std::string db_path;
    std::vector<FailurePattern> failure_patterns;
    std::unordered_map<std::string, std::tuple<double, double, std::string>> last_retrieval;
    double weights[3] = {0.6, 0.4, 0.0};
    double learning_rate = 0.02;
    std::unordered_map<std::string, int> consec_fail;
    void load_all() {
        load_memory();
        load_failures();
    }
    void load_memory() {
        fs::path mem_file = fs::path(db_path) / "semantic_memory.json";
        if (!fs::exists(mem_file)) return;
        try {
            json data = json::parse(std::ifstream(mem_file));
            for (auto &item : data) {
                auto entry = std::make_shared<SemanticMemoryEntry>();
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
                memory.push_back(entry);
                exact[entry->task_hash] = entry;
            }
            update_idf();
        } catch (const std::exception &e) {
            std::cerr << "[Memory] Load error: " << e.what() << "\n";
        }
    }
    void load_failures() {
        fs::path fail_file = fs::path(db_path) / "failure_patterns.json";
        if (!fs::exists(fail_file)) return;
        try {
            json data = json::parse(std::ifstream(fail_file));
            for (auto &item : data) {
                FailurePattern p;
                p.task_hash = item.value("task_hash", "");
                p.error_type = item.value("error_type", "");
                p.error_signature = item.value("error_signature", "");
                if (item.contains("fix_actions")) {
                    for (auto &a : item["fix_actions"]) p.fix_actions.push_back(a);
                }
                p.hint = item.value("hint", "");
                p.occurrence_count = item.value("occurrence_count", 1);
                p.last_seen = item.value("last_seen", 0.0);
                failure_patterns.push_back(p);
            }
        } catch (...) {}
    }
    void save_all() {
        save_memory();
        save_failures();
    }
    void save_memory() {
        json data = json::array();
        for (auto &e : memory) {
            json vec = json::object();
            for (auto &[term, w] : e->word_vector) vec[term] = w;
            json terms = json::array();
            for (auto &t : e->term_set) terms.push_back(t);
            data.push_back({
                {"task_hash", e->task_hash},
                {"task_text", e->task_text},
                {"solution_code", e->solution_code},
                {"timestamp", e->success_timestamp},
                {"api_calls_saved", e->api_calls_saved},
                {"access_count", e->access_count},
                {"last_access_time", e->last_access_time},
                {"vector", vec},
                {"terms", terms},
                {"norm", e->norm},
                {"is_plan", e->is_plan},
                {"success_count", e->success_count},
                {"failure_count", e->failure_count}
            });
        }
        std::ofstream(fs::path(db_path) / "semantic_memory.json") << data.dump(2);
    }
    void save_failures() {
        json data = json::array();
        for (auto &p : failure_patterns) {
            json fixes = json::array();
            for (auto &f : p.fix_actions) fixes.push_back(f);
            data.push_back({
                {"task_hash", p.task_hash},
                {"error_type", p.error_type},
                {"error_signature", p.error_signature},
                {"fix_actions", fixes},
                {"hint", p.hint},
                {"occurrence_count", p.occurrence_count},
                {"last_seen", p.last_seen}
            });
        }
        std::ofstream(fs::path(db_path) / "failure_patterns.json") << data.dump(2);
    }
    void update_idf() {
        total_docs = memory.size();
        std::unordered_map<std::string, int> doc_freq;
        std::unordered_map<std::string, std::unordered_map<std::string, int>> raw_tfs;
        for (auto &e : memory) {
            auto tokens = tokenize(e->task_text);
            std::unordered_map<std::string, int> freq;
            for (auto &t : tokens) freq[t]++;
            raw_tfs[e->task_hash] = freq;
            for (auto &[term, _] : freq) doc_freq[term]++;
        }
        idf.clear();
        for (auto &[term, df] : doc_freq)
            idf[term] = std::log(1.0 + total_docs / static_cast<double>(df));
        inverted.clear();
        for (auto &e : memory) {
            auto &freq = raw_tfs[e->task_hash];
            if (freq.empty()) {
                e->word_vector.clear();
                e->term_set.clear();
                e->term_count = 0;
                e->norm = 0.0;
                continue;
            }
            int max_freq = 0;
            for (auto &[_, c] : freq) max_freq = std::max(max_freq, c);
            std::vector<std::pair<std::string, double>> vec;
            for (auto &[term, count] : freq) {
                double weight = (static_cast<double>(count) / max_freq) * idf[term];
                vec.push_back({term, weight});
                inverted[term].push_back({e, weight});
            }
            std::sort(vec.begin(), vec.end());
            e->word_vector = vec;
            e->term_set.clear();
            for (auto &[term, _] : freq) e->term_set.insert(term);
            e->term_count = e->term_set.size();
            double norm = 0.0;
            for (auto &[_, w] : vec) norm += w * w;
            e->norm = norm;
        }
    }
    static double jaccard(const std::unordered_set<std::string> &s1,
                          const std::unordered_set<std::string> &s2) {
        if (s1.empty() || s2.empty()) return 0.0;
        int inter = 0;
        for (auto &x : s1) if (s2.count(x)) inter++;
        int uni = s1.size() + s2.size() - inter;
        return uni ? static_cast<double>(inter) / uni : 0.0;
    }
    double hybrid(double cos, double jac) const {
        cos = std::max(0.0, std::min(1.0, cos));
        jac = std::max(0.0, std::min(1.0, jac));
        double sum = weights[0] + weights[1];
        if (sum <= 1e-12) return 0.5 * (cos + jac);
        return std::max(0.0, std::min(1.0, (weights[0] * cos + weights[1] * jac) / sum));
    }
    double memory_reliability(const std::shared_ptr<SemanticMemoryEntry> &e) const {
        int total = e->success_count + e->failure_count;
        if (total <= 0) return 0.5;
        return (e->success_count + 1.0) / (total + 2.0);
    }
};

class FileManager {
public:
    explicit FileManager(const std::string &backup_dir) : backup_dir(backup_dir) {
        fs::create_directories(backup_dir);
    }
    std::string write_file(const std::string &file_path, const std::string &content) {
        fs::path path(file_path);
        if (path.has_parent_path()) fs::create_directories(path.parent_path());
        std::string fixed = content;
        fixed = std::regex_replace(fixed,
            std::regex(R"(#include\s*<json/json\.h>)"),
            "#include <nlohmann/json.hpp>");
        fs::path tmp = path.string() + ".tmp";
        {
            std::ofstream f(tmp);
            f << fixed;
        }
        fs::rename(tmp, path);
        return "File written successfully: " + file_path;
    }
    std::string read_file(const std::string &file_path) const {
        try {
            std::ifstream f(file_path);
            std::ostringstream ss;
            ss << f.rdbuf();
            return ss.str();
        } catch (const std::exception &e) {
            return "Error reading file: " + std::string(e.what());
        }
    }
    std::string backup_file(const std::string &file_path) const {
        fs::path path(file_path);
        if (!fs::exists(path)) return "";
        int next_ver = 1;
        for (auto &p : fs::directory_iterator(backup_dir)) {
            std::string name = p.path().filename().string();
            std::string prefix = path.filename().string() + "_v";
            if (name.rfind(prefix, 0) == 0 && name.find(".bak") != std::string::npos) {
                int v = std::atoi(name.substr(prefix.size()).c_str());
                next_ver = std::max(next_ver, v + 1);
            }
        }
        fs::path backup = fs::path(backup_dir) /
                          (path.filename().string() + "_v" + std::to_string(next_ver) + ".bak");
        fs::copy_file(path, backup, fs::copy_options::overwrite_existing);
        return backup.string();
    }

private:
    fs::path backup_dir;
};

class ToolExecutor {
public:
    ToolExecutor() {}

    std::string execute_action_dict(json action) {
        try {
            if (!action.is_object())
                return "Error: Tool action is not a JSON object.";

            action = apply_aliases(action);

            if (!action.contains("tool") || !action["tool"].is_string())
                return "Error: Tool action is missing a string 'tool' field.";

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

            if (!required.count(tool))
                return "Error: Tool '" + tool + "' is not recognized.";

            std::set<std::string> missing;
            for (auto &arg : required.at(tool))
                if (!args.contains(arg))
                    missing.insert(arg);

            if (!missing.empty()) {
                std::string msg;
                for (auto &m : missing)
                    msg += m + " ";
                return "Error: Missing required arguments for '" + tool + "': " + msg;
            }

            if (tool == "write_file") {
                std::string fp = get_string_arg(args, "file_path");
                std::string content = get_string_arg(args, "content");

                if (fp.empty())
                    return "Error: write_file 'file_path' must be a non-empty string.";
                if (content.empty())
                    return "Error: write_file 'content' must be a non-empty string.";

                std::string syntax = validate_content_syntax(fp, content);
                if (!syntax.empty() && syntax.find("FAILED") != std::string::npos) {
                    return "Error: " + syntax +
                           "\nFile was NOT written. Fix the syntax error and try again.";
                }

                return file_manager.write_file(fp, content);
            }

            if (tool == "read_file") {
                std::string fp = get_string_arg(args, "file_path");
                if (fp.empty())
                    return "Error: read_file 'file_path' must be a non-empty string.";
                return file_manager.read_file(fp);
            }

            if (tool == "execute_file") {
                std::string fp = get_string_arg(args, "file_path");
                if (fp.empty())
                    return "Error: execute_file 'file_path' must be a non-empty string.";

                std::string stdin_input = get_string_arg(args, "stdin");
                return executor.execute_file(fp, stdin_input);
            }

            if (tool == "make_directory") {
                std::string dir = get_string_arg(args, "dir_path");
                if (dir.empty())
                    return "Error: make_directory 'dir_path' must be a non-empty string.";

                fs::create_directories(dir);
                return "Directory created: " + dir;
            }

            if (tool == "install_package") {
                std::string pkg = get_string_arg(args, "package");
                if (pkg.empty())
                    return "Error: install_package 'package' must be a non-empty string.";

                return executor.run_shell_command("pip install " + pkg,
                                                  executor.max_install_timeout);
            }

            if (tool == "compile_cpp") {
                std::string fp = get_string_arg(args, "file_path");
                if (fp.empty())
                    return "Error: compile_cpp 'file_path' must be a non-empty string.";
                return executor.compile_cpp(fp);
            }

            if (tool == "run_shell_command") {
                std::string cmd = get_string_arg(args, "command");
                if (cmd.empty())
                    return "Error: run_shell_command 'command' must be a non-empty string.";
                return executor.run_shell_command(cmd);
            }

            if (tool == "finish")
                return "Task marked as finished. Verifying results...";

            return "Unknown tool";
        } catch (const std::exception &e) {
            return "Exception executing tool: " + std::string(e.what());
        }
    }

    std::vector<json> extract_actions(const std::string &text) {
        std::string clean = strip_code_fences(text);
        std::vector<json> objects;

        size_t idx = 0;
        while ((idx = clean.find('{', idx)) != std::string::npos) {
            size_t start = idx;
            int stack = 0;
            bool in_str = false, escape = false;
            size_t i = start;

            for (; i < clean.size(); ++i) {
                char ch = clean[i];
                if (escape) {
                    escape = false;
                } else if (ch == '\\') {
                    escape = true;
                } else if (ch == '"') {
                    in_str = !in_str;
                } else if (!in_str) {
                    if (ch == '{') stack++;
                    else if (ch == '}') {
                        stack--;
                        if (stack == 0) { i++; break; }
                    }
                }
            }

            if (stack == 0) {
                std::string block = clean.substr(start, i - start);
                try {
                    json parsed = json::parse(block);
                    if (parsed.is_object() && parsed.contains("tool"))
                        objects.push_back(parsed);
                } catch (...) {
                    std::string repaired = repair_json_fragment(block);
                    try {
                        json parsed = json::parse(repaired);
                        if (parsed.is_object() && parsed.contains("tool"))
                            objects.push_back(parsed);
                    } catch (...) {}
                }
                idx = i;
            } else {
                break;
            }
        }

        if (objects.empty()) {
            auto direct = extract_write_file_direct(clean);
            if (!direct.is_null()) objects.push_back(direct);
        }

        if (objects.empty()) {
            objects = extract_xml_tool_calls(clean);
        }

        std::vector<json> result;
        std::set<std::string> seen;

        for (auto &obj : objects) {
            std::string key = obj.dump();
            if (!seen.count(key)) {
                seen.insert(key);
                result.push_back(obj);
            }
        }

        return result;
    }

    CodeExecutor executor;
    FileManager file_manager{".agent_backups"};

private:
    static std::string get_string_arg(const json &args, const std::string &key) {
        if (!args.is_object() || !args.contains(key) || !args[key].is_string())
            return "";
        return args[key].get<std::string>();
    }

    std::string strip_code_fences(const std::string &text) {
        std::string clean = text;
        size_t pos = 0;

        while ((pos = clean.find("```")) != std::string::npos) {
            size_t end = clean.find("```", pos + 3);
            if (end == std::string::npos) {
                clean = clean.substr(0, pos);
                break;
            }

            std::string block = clean.substr(pos + 3, end - pos - 3);
            if (block.rfind("json", 0) == 0)
                block = block.substr(4);

            clean = clean.substr(0, pos) + block + clean.substr(end + 3);
        }

        return clean;
    }

    std::string repair_json_fragment(const std::string &fragment) {
        std::string result;
        bool in_string = false;
        bool escape_next = false;

        for (char ch : fragment) {
            if (escape_next) {
                result += ch;
                escape_next = false;
                continue;
            }
            if (ch == '\\') {
                result += ch;
                escape_next = true;
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
        std::regex trailing_comma(R"(,(\s*[}\]]))");
        result = std::regex_replace(result, trailing_comma, "$1");

        std::regex unquoted_key(R"(([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:)");
        result = std::regex_replace(result, unquoted_key, "$1\"$2\":");

        return result;
    }

    json extract_write_file_direct(const std::string &text) {
        std::regex tool_re(R"("tool"\s*:\s*"write_file")");
        std::smatch m;
        if (!std::regex_search(text, m, tool_re))
            return json::object();

        size_t obj_start = text.rfind('{', m.position());
        if (obj_start == std::string::npos)
            return json::object();

        int stack = 0;
        bool in_str = false, escape = false;
        size_t i = obj_start;

        for (; i < text.size(); ++i) {
            char ch = text[i];
            if (escape) {
                escape = false;
            } else if (ch == '\\') {
                escape = true;
            } else if (ch == '"') {
                in_str = !in_str;
            } else if (!in_str) {
                if (ch == '{') stack++;
                else if (ch == '}') {
                    stack--;
                    if (stack == 0) { i++; break; }
                }
            }
        }

        if (stack != 0)
            return json::object();

        std::string obj_str = text.substr(obj_start, i - obj_start);

        std::regex fp_re(R"re("file_path"\s*:\s*"((?:\\.|[^"\\])*)")re");
        std::smatch fp_m;
        if (!std::regex_search(obj_str, fp_m, fp_re))
            return json::object();
        std::string file_path = fp_m[1].str();

        std::regex cont_re(R"re("content"\s*:\s*"((?:\\.|[^"\\])*)")re");
        std::smatch cont_m;
        if (!std::regex_search(obj_str, cont_m, cont_re))
            return json::object();
        std::string content = cont_m[1].str();

        std::replace(content.begin(), content.end(), '\\', ' ');

        return {{"tool", "write_file"},
                {"args", {{"file_path", file_path}, {"content", content}}}};
    }

    std::vector<json> extract_xml_tool_calls(const std::string &text) {
        std::vector<json> actions;
        std::regex block_re(R"(<tool_call>(.*?)</tool_call>)");
        std::sregex_iterator it(text.begin(), text.end(), block_re), end;

        for (; it != end; ++it) {
            std::string block = it->str(1);
            std::smatch m;
            std::string tool;

            if (std::regex_search(block, m, std::regex(R"(<tool_name>(.*?)</tool_name>)")))
                tool = m[1].str();

            json args = json::object();
            std::regex key_re(R"(<arg_key>(.*?)</arg_key>)");
            std::regex val_re(R"(<arg_value>(.*?)</arg_value>)");
            std::sregex_iterator kit(block.begin(), block.end(), key_re);
            std::sregex_iterator vit(block.begin(), block.end(), val_re);
            std::sregex_iterator kend;

            for (; kit != kend && vit != std::sregex_iterator(); ++kit, ++vit)
                args[kit->str(1)] = vit->str(1);

            if (!tool.empty())
                actions.push_back({{"tool", tool}, {"args", args}});
        }

        return actions;
    }

    json apply_aliases(json action) {
        std::string tool = action.value("tool", "");

        if (!action.contains("args") && action.is_object()) {
            json args = json::object();
            for (auto &[k, v] : action.items()) {
                if (k != "tool")
                    args[k] = v;
            }
            if (!args.empty())
                action["args"] = args;
        }

        static const std::map<std::string, std::map<std::string, std::string>> aliases = {
            {"write_file", {{"path", "file_path"}, {"filename", "file_path"}, {"name", "file_path"}}},
            {"read_file", {{"path", "file_path"}, {"file", "file_path"}}},
            {"execute_file", {{"path", "file_path"}, {"file", "file_path"}}},
            {"compile_cpp", {{"path", "file_path"}, {"file", "file_path"}}},
            {"run_shell_command", {{"cmd", "command"}, {"shell_args", "command"}}},
            {"make_directory", {{"path", "dir_path"}, {"dir_name", "dir_path"},
                                {"name", "dir_path"}, {"directory", "dir_path"}}}
        };

        if (auto it = aliases.find(tool); it != aliases.end()) {
            if (action.contains("args") && action["args"].is_object()) {
                for (auto &[bad, good] : it->second) {
                    if (action["args"].contains(bad) && !action["args"].contains(good)) {
                        action["args"][good] = action["args"][bad];
                        action["args"].erase(bad);
                    }
                }
            }
        }

        return action;
    }

    std::string validate_content_syntax(const std::string &file_path,
                                        const std::string &content) {
        std::string ext = fs::path(file_path).extension().string();
        if (ext == ".py")
            return executor.validate_syntax(content, "python");
        if (ext == ".cpp" || ext == ".cc" || ext == ".cxx")
            return executor.validate_syntax(content, "cpp");
        return "";
    }
};

class TaskDecomposer {
public:
    explicit TaskDecomposer(LLMBackendManager &llm) : llm(llm) {}
    std::vector<std::string> decompose(const std::string &task) {
        json schema = {{"checklist", json::array({"sub-goal 1"})}};
        std::string prompt = "Break this task into 3-5 sequential sub-goals. Output ONLY JSON.\nTask: " + task;
        json res = llm.structured_chat({{"user", prompt}}, schema, 0.0, 256);
        if (res.contains("checklist") && res["checklist"].is_array()) {
            std::vector<std::string> goals;
            for (auto &g : res["checklist"]) goals.push_back(g.get<std::string>());
            return goals;
        }
        return {task};
    }
private:
    LLMBackendManager &llm;
};

class DependencyPlanner {
public:
    explicit DependencyPlanner(LLMBackendManager &llm) : llm(llm) {}

    std::vector<json> generate_plan(const std::string &task,
                                    const std::vector<std::string> &expected_outputs) {
        if (expected_outputs.empty()) return {};

        std::string tools_desc =
            "- write_file: file_path, content\n"
            "- read_file: file_path\n"
            "- execute_file: file_path, stdin\n"
            "- compile_cpp: file_path\n"
            "- run_shell_command: command\n"
            "- make_directory: dir_path\n"
            "- finish: none\n";

        std::string prompt =
            "Task: " + task + "\n"
            "Required output files: " + join(expected_outputs, ", ") + "\n\n"
            "Available tools:\n" + tools_desc + "\n"
            "Create a JSON plan with a list of actions that will accomplish the task. "
            "Always create directories before writing files, and execute scripts that produce output files.\n"
            "Output ONLY the JSON array of actions.";

        json schema = {{"actions", json::array({
            {{"tool", "write_file (or make_directory, etc.)"},
             {"args", {{"any key", "value"}}}}
        })}};

        json plan = llm.structured_chat({{"user", prompt}}, schema, 0.0, 1024);

        std::vector<json> raw = extract_plan_actions(plan);
        std::vector<json> actions;
        actions.reserve(raw.size());

        for (const auto &a : raw) {
            json norm = normalize_action(a);
            if (!norm.is_null() && norm.is_object() && norm.contains("tool"))
                actions.push_back(std::move(norm));
        }

        if (actions.empty()) return fallback_plan(expected_outputs);
        return fix_files(actions, expected_outputs);
    }

private:
    LLMBackendManager &llm;

    static std::string join(const std::vector<std::string> &v, const std::string &sep) {
        std::string out;
        for (size_t i = 0; i < v.size(); ++i) {
            if (i) out += sep;
            out += v[i];
        }
        return out;
    }

    static void collect_actions(const json &node, std::vector<json> &out) {
        if (node.is_array()) {
            for (const auto &child : node)
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

    static std::vector<json> extract_plan_actions(const json &plan) {
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

    static json normalize_action(const json &a) {
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

        // If the LLM accidentally places action arguments at the top level,
        // fold them into the "args" object where execute_action_dict expects them.
        for (auto &[k, v] : action.items()) {
            if (k != "tool" && k != "args")
                action["args"][k] = v;
        }

        return action;
    }

    std::vector<json> fallback_plan(const std::vector<std::string> &files) {
        std::vector<json> actions;
        std::set<std::string> seen_dirs;

        for (auto &fp : files) {
            fs::path p(fp);
            std::string parent = p.parent_path().string();
            if (!parent.empty() && parent != "." && !seen_dirs.count(parent)) {
                actions.push_back({{"tool", "make_directory"}, {"args", {{"dir_path", parent}}}});
                seen_dirs.insert(parent);
            }
        }

        return actions;
    }

    std::vector<json> fix_files(std::vector<json> actions,
                                const std::vector<std::string> &expected) {
        std::set<std::string> names;
        for (auto &p : expected) names.insert(fs::path(p).filename().string());

        for (auto &act : actions) {
            if (act.value("tool", "") == "write_file") {
                std::string fp = act["args"].value("file_path", "");
                if (!fp.empty() && names.count(fs::path(fp).filename().string())) {
                    for (auto &real : expected) {
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

class CodingAgent {
public:   
    CodingAgent()
	    : semantic_memory(),
	      llm({
	          {{"groq", "llama-3.3-70b-versatile"},
	           {"openrouter", "meta-llama/llama-3.3-70b-instruct"},
	           {"google", "gemini-3.1-flash-lite"}},
	          {"groq", "openrouter", "google"},
	          30
	      }),
	      //llm({
	          //{{"groq", "qwen/qwen3-32b"},
	           //{"openrouter", "cohere/north-mini-code:free"},
	           //{"google", "gemini-3.5-flash"}},
	          //{"groq", "openrouter", "google"},
	          //30
	      //}),
	      tool_executor(),
	      decomposer(llm),
	      planner(llm) {}      
    std::string run(const std::string &task,
                    int max_turns = 6,
                    const std::vector<std::string> &expected_outputs = {},
                    const std::vector<json> &validation_checks = {}) {
        current_task = task;
        this->expected_outputs = expected_outputs;
        this->validation_checks = validation_checks;
        session_context.clear();
        last_validation_feedback.clear();
        last_raw_error.clear();
        written_files.clear();
        successful_plan_actions.clear();
        aborted = false;
        std::string task_hash = md5_hash(task);
        std::string cached = semantic_memory.retrieve_similar(task);
        if (!cached.empty()) {
            std::vector<json> plan = extract_replayable_plan(cached);
            if (!plan.empty()) {
                auto [score, missing] = score_plan_against_validation(plan);
                if (score >= 0.4) {
                    std::cout << "[Agent] Replaying cached plan (score=" << score << ")\n";
                    if (score < 0.95) plan = repair_plan_suffix(plan, missing, task);
                    successful_plan_actions = plan;
                    execute_plan(plan);
                    std::string early = check_early_success();
                    if (!early.empty()) return early;
                }
            }
        }
        std::cout << "[Agent] Generating plan...\n";
        auto sub_goals = decomposer.decompose(task);
        if (sub_goals.size() > 1) {
            std::cout << "[Agent] Task decomposed into " << sub_goals.size() << " sub-goals.\n";
            session_context.push_back("Strategic Plan Breakdown: " + join(sub_goals, ", "));
        }
        std::vector<json> plan = planner.generate_plan(task, expected_outputs);
        if (!plan.empty()) {
            successful_plan_actions = plan;
            execute_plan(plan);
            std::string early = check_early_success();
            if (!early.empty()) return early;
        }
        std::string file_exists_hint = build_file_exists_hint();
        for (int turn = 0; turn < max_turns; ++turn) {
            if (aborted) return "FAILED: Aborted due to timeout or external signal.";
            if (!last_raw_error.empty()) {
                std::string hint = semantic_memory.get_hint_for_error(last_raw_error);
                if (!hint.empty()) session_context.push_back("[FailureHint] " + hint);
                std::string local = get_error_hint(last_raw_error);
                if (!local.empty()) session_context.push_back(local);
            }
            std::string progress = build_progress_hint();
            std::string validation_reqs = build_validation_requirements();
            std::string history = join_last(session_context, 8);
            std::string warning = last_validation_feedback.empty()
                ? "" : "\n🚨 VALIDATION FAILED 🚨\n" + last_validation_feedback + "\n";
            std::string user_msg =
                "TASK: " + task + "\n" +
                file_exists_hint +
                validation_reqs + "\n" +
                progress + "\n" +
                warning +
                "HISTORY:\n" + history + "\n" +
                "──────────────────────────────────────\n" +
                "Turn " + std::to_string(turn + 1) + "/" + std::to_string(max_turns) +
                " - What is your NEXT action?";
            std::cout << "\n[Turn " << turn + 1 << "] Querying LLM...\n";
            ChatResult r = llm.chat({{"system", system_prompt()}, {"user", user_msg}}, 0.0, 4096);
            if (!r.ok) {
                std::cout << "[Turn " << turn + 1 << "] ❌ LLM Error: " << r.error << "\n";
                session_context.push_back("Turn " + std::to_string(turn + 1) + " LLM Error: " + r.error);
                continue;
            }
            std::cout << "[Turn " << turn + 1 << "] LLM response (first 300 chars): "
                      << r.content.substr(0, 300) << "\n";
            auto actions = tool_executor.extract_actions(r.content);
            if (actions.empty()) {
                std::cout << "[Turn " << turn + 1 << "] ⚠️  No valid action extracted\n";
                session_context.push_back("Turn " + std::to_string(turn + 1) + ": No valid action found.");
                continue;
            }
            bool halted = false, finish_called = false;
            for (size_t ai = 0; ai < actions.size(); ++ai) {
                if (aborted) return "FAILED: Aborted.";
                std::string tool_name = actions[ai].value("tool", "unknown");
                std::cout << "[Turn " << turn + 1 << "] 🔧 Action " << ai + 1 << "/"
                          << actions.size() << ": " << tool_name << "\n";
                actions[ai] = fix_action_paths(actions[ai]);
                std::string observation = tool_executor.execute_action_dict(actions[ai]);
                std::cout << "[Turn " << turn + 1 << "] 📤 Observation (first 200 chars): "
                          << observation.substr(0, 200) << "\n";
                session_context.push_back("Turn " + std::to_string(turn + 1) + "." +
                                          std::to_string(ai + 1) + ": " + tool_name +
                                          "\n→ " + observation.substr(0, 500));
                successful_plan_actions.push_back(actions[ai]);
                if (is_critical_error(observation)) {
                    last_raw_error = observation;
                    semantic_memory.store_failure(task_hash,
                        error_type_from_output(observation), observation, {},
                        get_error_hint(observation));
                    std::cout << "[Turn " << turn + 1 << "] 🛑 Critical error, halting batch.\n";
                    halted = true;
                    break;
                }
                if (observation.find("Return code:") != std::string::npos &&
                    observation.find("Return code: 0") == std::string::npos) {
                    last_raw_error = observation;
                    semantic_memory.store_failure(task_hash,
                        error_type_from_output(observation), observation, {},
                        get_error_hint(observation));
                }
                if (tool_name == "write_file") {
                    std::string fp = actions[ai]["args"].value("file_path", "");
                    if (!fp.empty() && std::find(written_files.begin(), written_files.end(), fp) == written_files.end())
                        written_files.push_back(fp);
                    std::cout << "[Turn " << turn + 1 << "] 📝 Wrote: " << fp << "\n";
                }
                if (tool_name == "finish") {
                    finish_called = true;
                    halted = true;
                    break;
                }
            }
            if (finish_called) {
                std::cout << "[Turn " << turn + 1 << "] 🏁 Finish called, attempting validation...\n";
                attempt_direct_execution();
            }
            if (!halted || finish_called) {
                std::string early = check_early_success();
                if (!early.empty()) return early;
            }
            if (finish_called) {
                std::vector<std::string> missing_files;
                for (auto &f : expected_outputs)
                    if (!fs::exists(f)) missing_files.push_back("❌ File not found: " + f);
                std::vector<std::string> missing_checks;
                for (auto &check : validation_checks) {
                    std::string type = check.value("type", "");
                    if (type == "file_exists") {
                        std::string p = check.value("path", "");
                        if (!fs::exists(p)) missing_checks.push_back("❌ File not found: " + p);
                    } else if (type == "execution") {
                        std::string fp = check.value("file", "");
                        if (!fs::exists(fp)) {
                            missing_checks.push_back("❌ Script not found: " + fp);
                        } else {
                            std::string result = tool_executor.executor.execute_file(
                                fp, check.value("input", ""));
                            if (result.find("Return code: 0") == std::string::npos)
                                missing_checks.push_back("❌ Script " + fp + " failed (RC != 0): " +
                                                         result.substr(0, 150));
                            else if (check.contains("expect")) {
                                std::string expect = check["expect"];
                                std::string low_res = result;
                                std::transform(low_res.begin(), low_res.end(), low_res.begin(), ::tolower);
                                std::string low_exp = expect;
                                std::transform(low_exp.begin(), low_exp.end(), low_exp.begin(), ::tolower);
                                if (low_res.find(low_exp) == std::string::npos)
                                    missing_checks.push_back("❌ Script " + fp + " output missing '" +
                                                             expect + "': " + result.substr(0, 150));
                            }
                        }
                    }
                }
                std::vector<std::string> parts;
                if (!missing_files.empty()) parts.push_back("Missing output files: " + join(missing_files, ", "));
                if (!missing_checks.empty()) parts.push_back(join(missing_checks, "\n"));
                last_validation_feedback = parts.empty()
                    ? "Unknown validation failure."
                    : join(parts, "\n");
                std::cout << "[Turn " << turn + 1 << "] ❌ VALIDATION FAILED:\n"
                          << last_validation_feedback << "\n";
                session_context.push_back("Turn " + std::to_string(turn + 1) +
                                          ": finish FAILED. " + last_validation_feedback);
            }
        }
        std::cout << "\n❌ MAX TURNS REACHED (" << max_turns << ")\n";
        if (!last_validation_feedback.empty() || !last_raw_error.empty()) {
            std::string err = !last_validation_feedback.empty() ? last_validation_feedback : last_raw_error;
            semantic_memory.store_failure(task_hash, error_type_from_output(err), err, {},
                "Task failed after max turns. Review validation requirements.");
        }
        semantic_memory.report_outcome(task_hash, false);
        return "FAILED: Maximum turns reached.";
    }
    void abort() { aborted = true; llm.abort(); }
    void reset_abort() { aborted = false; llm.reset_abort(); llm.reset_cooldowns(); }
    SemanticMemoryManager semantic_memory;
    LLMBackendManager llm;

private:
    ToolExecutor tool_executor;
    TaskDecomposer decomposer;
    DependencyPlanner planner;
    std::vector<std::string> session_context;
    std::string last_validation_feedback;
    std::string last_raw_error;
    std::vector<std::string> written_files;
    std::vector<std::string> expected_outputs;
    std::vector<json> validation_checks;
    std::string current_task;
    std::vector<json> successful_plan_actions;
    std::atomic<bool> aborted{false};
	static std::string system_prompt() {
	    return
	        "You are a Senior AI Coding Agent specializing in algorithmic optimization.\n"
	        "Rules:\n"
	        "1. Communicate exclusively through JSON tool calls.\n"
	        "2. Format: {\"tool\": \"name\", \"args\": {\"arg\": \"val\"}}\n"
	        "3. You MAY output multiple JSON actions in a single response, separated by newlines.\n"
	        "4. Available tools (use exactly these argument names):\n"
	        "   - write_file: file_path, content\n"
	        "   - read_file: file_path\n"
	        "   - execute_file: file_path, stdin\n"
	        "   - compile_cpp: file_path\n"
	        "   - run_shell_command: command\n"
	        "   - make_directory: dir_path\n"
	        "   - finish: (no args)\n"
	        "5. For file creation tasks: write_file → execute_file → verify → finish.\n"
	        "6. NEVER call finish until you have EXECUTED any script that creates output files.\n"
	        "7. If validation fails, read the feedback and fix the exact issue mentioned.\n"
	        "8. Always implement complete, functional code without placeholders or TODO.\n"
	        "9. Pay close attention to VALIDATION REQUIREMENTS.\n"
	        "10. CRITICAL: Python multi-line strings MUST use triple quotes or explicit \\n escapes.\n"
	        "11. CRITICAL: Inside JSON string values, escape all double quotes with backslash.\n"
	        "12. CRITICAL: Write files to the EXACT paths specified in EXPECTED OUTPUT FILES.\n"
	        "13. CRITICAL: For C++ JSON output, use ONLY <nlohmann/json.hpp>.\n"
	        "14. CRITICAL: If a VALIDATION REQUIREMENT lists an 'expected substring', "
	        "your program MUST print that exact substring to stdout during normal execution. "
	        "If it would not naturally appear, add an explicit print statement, e.g. "
	        "print('power') or std::cout << \"CycleError\" << std::endl;. "
	        "Comments and source code do NOT count; the substring must appear in runtime output.\n";
	}
    std::string get_error_hint(const std::string &error_output) const {
        std::string low = error_output;
        std::transform(low.begin(), low.end(), low.begin(), ::tolower);
        if (low.find("unterminated string") != std::string::npos)
            return "HINT: Unterminated string literal. Use triple quotes or \\n.";
        if (low.find("indentationerror") != std::string::npos)
            return "HINT: Python indentation error. Use exactly 4 spaces per level.";
        if (low.find("modulenotfounderror") != std::string::npos ||
            low.find("no module named") != std::string::npos)
            return "HINT: Missing Python module. Install it or use standard library.";
        if (low.find("no such file") != std::string::npos)
            return "HINT: File not found. Ensure write_file succeeds before execute_file.";
        if (low.find("json/json.h") != std::string::npos)
            return "HINT: Replace <json/json.h> with <nlohmann/json.hpp>.";
        return "";
    }
    json fix_action_paths(json action) {
	    if (!action.is_object())
	        return action;
	    std::string tool = action.value("tool", "");
	    if (tool != "write_file" && tool != "execute_file" &&
	        tool != "compile_cpp" && tool != "read_file")
	        return action;
	    if (!action.contains("args") || !action["args"].is_object())
	        return action;
	    if (!action["args"].contains("file_path") ||
	        !action["args"]["file_path"].is_string())
	        return action;
	    if (expected_outputs.empty())
	        return action;
	    std::string fp = action["args"]["file_path"].get<std::string>();
	    std::string basename = fs::path(fp).filename().string();
	    for (auto &expected : expected_outputs) {
	        if (fs::path(expected).filename() == basename && fp != expected) {
	            std::cout << "[Agent] Redirecting " << tool << " path: "
	                      << fp << " -> " << expected << "\n";
	            action["args"]["file_path"] = expected;
	            break;
	        }
	    }
	    return action;
	}
    std::string build_file_exists_hint() const {
	    if (expected_outputs.empty() && validation_checks.empty()) return "";
	    std::string result = "📁 EXPECTED OUTPUT FILES:\n";
	    for (auto &fp : expected_outputs)
	        result += std::string(fs::exists(fp) ? "  ✅ " : "  ❌ ") + fp + "\n";
	    for (auto &check : validation_checks) {
	        if (check.value("type", "") == "file_exists") {
	            std::string p = check.value("path", "");
	            result += std::string(fs::exists(p) ? "  ✅ " : "  ❌ ") + "File exists: " + p + "\n";
	        } else if (check.value("type", "") == "execution") {
	            std::string fp = check.value("file", "");
	            result += std::string(fs::exists(fp) ? "  ✅ " : "  ❌ ") + "Script ready: " + fp + "\n";
	        }
	    }
	    return result;
	}
    std::string build_progress_hint() const {
        if (expected_outputs.empty() && validation_checks.empty()) return "";
        std::vector<std::string> lines = {"📋 PROGRESS CHECKLIST:"};
        for (auto &fp : expected_outputs) {
            lines.push_back(std::string(fs::exists(fp) ? "  ✅ " : "  ❌ ") + fp);
        }
        for (auto &check : validation_checks) {
            if (check.value("type", "") == "file_exists") {
                std::string p = check.value("path", "");
                lines.push_back(std::string(fs::exists(p) ? "  ✅ " : "  ❌ ") + "File exists: " + p);
            } else if (check.value("type", "") == "execution") {
                std::string fp = check.value("file", "");
                lines.push_back(std::string(fs::exists(fp) ? "  ✅ " : "  ❌ ") + "Script ready: " + fp);
                if (check.contains("expect"))
                    lines.push_back("     🔍 REQUIRED OUTPUT SUBSTRING: '" +
                                    check["expect"].get<std::string>() + "' (case-insensitive)");
                if (check.contains("input"))
                    lines.push_back("     📥 STDIN INPUT: " + check["input"].dump());
            }
        }
        std::vector<std::string> missing;
        for (auto &fp : expected_outputs)
            if (!fs::exists(fp)) missing.push_back(fp);
        if (!missing.empty()) {
            lines.push_back("\n➡️  NEXT: Create these missing files: " + join(missing, ", "));
        } else {
            lines.push_back("\n✅ All expected files exist. Execute validation scripts and call finish.");
        }
        return join(lines, "\n");
    }
	std::string build_validation_requirements() const {
	    if (validation_checks.empty()) return "";
	    std::vector<std::string> lines = {
	        "\n🔒 VALIDATION REQUIREMENTS — YOUR CODE MUST SATISFY ALL OF THESE:"
	    };
	    bool has_expect = false;
	    for (auto &check : validation_checks) {
	        if (check.value("type", "") == "file_exists") {
	            lines.push_back("  • File MUST exist: " + check.value("path", ""));
	        } else if (check.value("type", "") == "execution") {
	            lines.push_back("  • Script " + check.value("file", "") +
	                            " MUST run with exit code 0");
	            if (check.contains("input"))
	                lines.push_back("    Stdin provided: " + check["input"].dump());
	            if (check.contains("expect")) {
	                lines.push_back("    Output MUST contain EXACTLY: '" +
	                                check["expect"].get<std::string>() + "'");
	                has_expect = true;
	            }
	        }
	    }
	    if (has_expect) {
	        lines.push_back(
	            "\n🚨 MARKER RULE: If any required substring above would not naturally "
	            "appear in the program's stdout, you MUST add an explicit print statement "
	            "for it. Examples: print('power'), print('CycleError'), "
	            "std::cout << \"null\" << std::endl;. The substring must be visible in "
	            "the runtime output; comments and source code do not count.");
	    }
	    return join(lines, "\n");
	}
    void execute_plan(const std::vector<json> &plan) {
        for (size_t i = 0; i < plan.size(); ++i) {
            json action = fix_action_paths(plan[i]);
            std::string obs = tool_executor.execute_action_dict(action);
            std::string tool = action.value("tool", "unknown");
            session_context.push_back("Plan Step " + std::to_string(i + 1) + ": " +
                                      tool + " → " + obs.substr(0, 300));
            if (tool == "write_file") {
                std::string fp = action["args"].value("file_path", "");
                if (!fp.empty() && std::find(written_files.begin(), written_files.end(), fp) == written_files.end())
                    written_files.push_back(fp);
            }
        }
    }
    void attempt_direct_execution() {
        if (validation_checks.empty()) return;
        std::set<std::string> needed;
        for (auto &c : validation_checks) {
            if (c.value("type", "") == "file_exists") needed.insert(c.value("path", ""));
            else if (c.value("type", "") == "execution") needed.insert(c.value("file", ""));
        }
        std::vector<std::string> missing;
        for (auto &f : needed) if (!f.empty() && !fs::exists(f)) missing.push_back(f);
        if (missing.empty()) return;
        std::string target;
        std::string target_stdin;
        for (auto it = written_files.rbegin(); it != written_files.rend(); ++it) {
            if (!fs::exists(*it)) continue;
            if (it->ends_with(".py")) {
                target = *it;
                for (auto &c : validation_checks)
                    if (c.value("type", "") == "execution" && c.value("file", "") == *it)
                        target_stdin = c.value("input", "");
                break;
            } else if (it->ends_with(".cpp") || it->ends_with(".cc") || it->ends_with(".cxx")) {
                std::string compiled = tool_executor.executor.compile_cpp(*it);
                if (compiled.find("COMPILATION_FAILED") == std::string::npos) {
                    target = compiled;
                    break;
                }
            }
        }
        if (!target.empty()) {
            std::cout << "[Agent] 🔧 Auto-executing " << target
                      << " to create missing files\n";
            std::string result = tool_executor.executor.execute_file(target, target_stdin);
            session_context.push_back("[Auto-exec] Executed " + target +
                                      "\nResult: " + result.substr(0, 400));
        }
    }
    std::string check_early_success() {
        if (!validator()) return "";
        std::cout << "[Agent] ✅ Validation passed — terminating early.\n";
        json solution;
        solution["plan_actions"] = successful_plan_actions;
        json files = json::object();
        for (auto &fp : written_files) {
            if (fs::exists(fp))
                files[fp] = tool_executor.file_manager.read_file(fp);
        }
        solution["file_contents"] = files;
        semantic_memory.store(current_task, solution.dump(2));
        semantic_memory.report_outcome(md5_hash(current_task), true);
        return "SUCCESS: Task completed and validated.";
    }
    bool validator() {
        for (auto &check : validation_checks) {
            if (check.value("type", "") == "file_exists") {
                if (!fs::exists(check.value("path", ""))) return false;
            } else if (check.value("type", "") == "execution") {
                std::string fp = check.value("file", "");
                if (!fs::exists(fp)) return false;
                std::string result = tool_executor.executor.execute_file(fp, check.value("input", ""));
                if (result.find("Return code: 0") == std::string::npos) return false;
                if (check.contains("expect")) {
                    std::string expect = check["expect"];
                    std::string low_res = result;
                    std::transform(low_res.begin(), low_res.end(), low_res.begin(), ::tolower);
                    std::string low_exp = expect;
                    std::transform(low_exp.begin(), low_exp.end(), low_exp.begin(), ::tolower);
                    if (low_res.find(low_exp) == std::string::npos) return false;
                }
            }
        }
        for (auto &f : expected_outputs)
            if (!fs::exists(f)) return false;
        return true;
    }
    bool is_critical_error(const std::string &observation) const {
        return observation.find("Error: Syntax validation FAILED") != std::string::npos ||
               observation.find("COMPILATION_FAILED") != std::string::npos ||
               observation.find("Error: Tool") != std::string::npos ||
               observation.find("Exception executing tool") != std::string::npos;
    }
    static std::vector<json> extract_replayable_plan(const std::string &cached_solution) {
        try {
            json parsed = json::parse(cached_solution);
            if (parsed.is_object() && parsed.contains("plan_actions") &&
                parsed["plan_actions"].is_array()) {
                std::vector<json> plan;
                for (auto &a : parsed["plan_actions"])
                    if (a.is_object() && a.contains("tool")) plan.push_back(a);
                return plan;
            }
            if (parsed.is_array()) {
                std::vector<json> plan;
                for (auto &a : parsed)
                    if (a.is_object() && a.contains("tool")) plan.push_back(a);
                return plan;
            }
        } catch (...) {}
        return {};
    }
    std::pair<double, std::vector<std::string>> score_plan_against_validation(
        const std::vector<json> &plan) {
        std::set<std::string> written;
        for (auto &a : plan)
            if (a.value("tool", "") == "write_file")
                written.insert(a["args"].value("file_path", ""));
        std::vector<std::string> missing;
        for (auto &f : expected_outputs)
            if (!written.count(f) && !fs::exists(f)) missing.push_back(f);
        for (auto &c : validation_checks) {
            if (c.value("type", "") == "file_exists") {
                std::string p = c.value("path", "");
                if (!written.count(p) && !fs::exists(p)) missing.push_back(p);
            } else if (c.value("type", "") == "execution") {
                std::string fp = c.value("file", "");
                if (!written.count(fp) && !fs::exists(fp)) missing.push_back(fp);
            }
        }
        int total = std::max(1, static_cast<int>(expected_outputs.size() + validation_checks.size()));
        double score = 1.0 - static_cast<double>(missing.size()) / total;
        return {score, missing};
    }
    std::vector<json> repair_plan_suffix(const std::vector<json> &partial,
                                         const std::vector<std::string> &missing,
                                         const std::string &task) {
        if (missing.empty()) return partial;
        json schema = {{"actions", json::array({
            {{"tool", "write_file (or execute_file / finish / ...)"},
             {"args", {{"any", "value"}}}}
        })}};
        std::string prompt =
            "Task: " + task + "\n"
            "The following files / checks are still missing: " + join(missing, ", ") + "\n"
            "Existing successful prefix (do NOT repeat these):\n" +
            json(partial).dump(2) + "\n\n"
            "Output ONLY a JSON list of the additional actions needed to create the missing items "
            "and reach a successful finish. Prefer the shortest correct suffix.";
        json repaired = llm.structured_chat({{"user", prompt}}, schema, 0.0, 1024);
        if (repaired.contains("actions") && repaired["actions"].is_array()) {
            std::vector<json> result = partial;
            for (auto &a : repaired["actions"]) {
                std::string fp = a["args"].value("file_path", a["args"].value("file", ""));
                if (fp.empty() || a.value("tool", "") == "finish" ||
                    a.value("tool", "") == "execute_file" ||
                    std::any_of(missing.begin(), missing.end(),
                                [&](const std::string &m) {
                                    return fs::path(m).filename() == fs::path(fp).filename();
                                })) {
                    result.push_back(a);
                }
            }
            return result;
        }
        return partial;
    }
    static std::string join(const std::vector<std::string> &v, const std::string &sep) {
        std::string out;
        for (size_t i = 0; i < v.size(); ++i) {
            if (i) out += sep;
            out += v[i];
        }
        return out;
    }
    static std::string join_last(const std::vector<std::string> &v, size_t count) {
        size_t start = v.size() > count ? v.size() - count : 0;
        std::vector<std::string> sub(v.begin() + start, v.end());
        return join(sub, "\n");
    }
};

struct TestCase {
    std::string id, name, description, task_prompt;
    std::vector<std::string> expected_outputs;
    std::vector<json> validation_checks;
    int max_turns;
    int timeout;
};

class TestSuite {
public:
    TestSuite() { cases = default_cases(); }

    const std::vector<TestCase>& get_cases() const { return cases; }

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

        return cases;
    }
};

static void run_single_task(CodingAgent &agent) {
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cout << "Enter task description: ";
    std::string task;
    std::getline(std::cin, task);
    if (task.empty()) return;
    std::cout << "Expected output files (comma-separated, optional): ";
    std::string expected_line;
    std::getline(std::cin, expected_line);
    std::vector<std::string> expected_outputs;
    size_t pos = 0;
    while ((pos = expected_line.find(',')) != std::string::npos) {
        std::string f = expected_line.substr(0, pos);
        if (!f.empty()) expected_outputs.push_back(f);
        expected_line.erase(0, pos + 1);
    }
    if (!expected_line.empty()) expected_outputs.push_back(expected_line);
    std::vector<json> checks;
    if (!expected_outputs.empty()) {
        std::cout << "Add validation checks? (y/n): ";
        std::string add;
        std::getline(std::cin, add);
        if (add == "y" || add == "yes") {
            for (auto &fp : expected_outputs) {
                if (fp.ends_with(".py") || fp.ends_with(".cpp")) {
                    std::cout << "Expected output substring for " << fp << " (optional): ";
                    std::string expect;
                    std::getline(std::cin, expect);
                    std::cout << "Stdin input for " << fp << " (optional): ";
                    std::string stdin_str;
                    std::getline(std::cin, stdin_str);
                    json check = {{"type", "execution"}, {"file", fp}};
                    if (!expect.empty()) check["expect"] = expect;
                    if (!stdin_str.empty()) check["input"] = stdin_str;
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
    int max_turns = mt.empty() ? 6 : std::stoi(mt);
    std::string result = agent.run(task, max_turns, expected_outputs, checks);
    std::cout << "\nResult: " << result << "\n";
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
            std::cout << "2. Run tests\n";
            std::cout << "3. Stats\n";
            std::cout << "4. Quit\n";
            std::cout << "Choice: ";
            std::string choice;
            std::getline(std::cin, choice);
            if (choice == "4" || choice == "quit" || choice == "exit") {
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
                    if (!ids.empty()) test_ids.push_back(ids);
                }
                int total_tests = 0;
                int passed_tests = 0;
                for (auto &tc : suite.get_cases()) {
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
                            output = agent.run(tc.task_prompt, tc.max_turns,
                                               tc.expected_outputs, tc.validation_checks);
                        } catch (...) {
                            exc = std::current_exception();
                        }
                        done = true;
                    });
                    auto deadline = std::chrono::steady_clock::now() +
                                    std::chrono::seconds(tc.timeout);
                    while (!done && std::chrono::steady_clock::now() < deadline) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    }
                    double dur = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - start).count();
                    if (!done) {
                        std::cout << "✗ Timed out after " << tc.timeout << "s\n";
                        agent.abort();
                        t.join();
                    } else {
                        t.join();
                        if (exc) {
                            std::cout << "✗ Exception: ";
                            try {
                                std::rethrow_exception(exc);
                            } catch (const std::exception &e) {
                                std::cout << e.what();
                            } catch (...) {
                                std::cout << "unknown";
                            }
                            std::cout << "\n";
                        }   
                        else if (output.find("SUCCESS") != std::string::npos) {
                            std::cout << "✓ Passed in " << dur << "s\n";
                            ++passed_tests;
                        } else {
                            std::cout << "✗ Failed in " << dur << "s\n";
                        }
                    }
                }
                if (total_tests > 0) {
                    double pass_rate = 100.0 * passed_tests / total_tests;
                    std::cout << "\n================ TEST SUMMARY ================\n";
                    std::cout << "Passed: " << passed_tests << "/" << total_tests << "\n";
                    std::cout << "Pass rate: " << pass_rate << "%\n";
                } else {
                    std::cout << "\nNo matching tests were run.\n";
                }
            } else if (choice == "3") {
                agent.semantic_memory.print_stats();
            } else {
                std::cout << "Invalid choice.\n";
            }
        }
    } catch (const std::exception &e) {
        std::cerr << "Initialization error: " << e.what() << "\n";
        std::cerr << "Make sure at least one API key is set:\n";
        std::cerr << "  GROQ_API_KEY, OPENROUTER_API_KEY, or GOOGLE_API_KEY\n";
        return 1;
    }
    curl_global_cleanup();
    return 0;
}
