#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <deque>
#include <memory>
#include <functional>
#include <algorithm>
#include <regex>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <cstdlib>
#include <cstring>
#include <curl/curl.h>
#include <openssl/md5.h>
#include <nlohmann/json.hpp>
#include <unistd.h>
#include <thread>
#include <iterator>
#include <cmath>
#include <set>
#include <optional>
#include <array>

namespace fs = std::filesystem;
using json = nlohmann::json;

struct AgentConfig {
    std::string model_id = "llama-3.3-70b-versatile";
    std::string backup_dir = ".agent_backups";
    std::string memory_db_path = "./agent_memory_db";
    std::string memory_collection_name = "coding_agent_memory_v3";
    int max_execution_timeout = 30;
    int max_code_timeout = 15;
    int max_install_timeout = 60;
    float semantic_similarity_threshold = 0.80f;
};

struct EvolutionConfig {
    bool enabled = true;
    bool self_modify = false;
    bool auto_reflect = true;
    std::string test_suite_path = "./agent_test_suite";
    int max_context_turns = 6;
    std::string capability_registry_file = "./capabilities.json";
    double min_fitness_threshold = 0.8;
};

struct ValidationCheck {
    std::string type;
    std::string path;
    std::string file;
    std::string expect;
};

struct TestCase {
    std::string id;
    std::string name;
    std::string description;
    std::string task_prompt;
    std::vector<std::string> expected_outputs;
    std::vector<ValidationCheck> validation_checks;
    int max_turns = 15;
    int timeout = 120;
};

struct AnalysisResult {
    double timestamp;
    std::string task_hash;
    bool success;
    int turn_count;
    std::string error_type;
    std::string bottleneck;
    std::string suggested_capability;
};

struct ToolInfo {
    std::string description;
    std::string code;
    std::string source_task;
    double created;
    int uses = 0;
};

struct TestResult {
    std::string id;
    std::string name;
    bool success;
    bool llm_claimed_success;
    bool validation_passed;
    double duration;
    std::vector<std::string> details;
    std::string output_snippet;
    std::string error;
    std::string traceback;
};

struct SemanticMemoryEntry {
    std::string task_hash;
    std::string task_text;
    std::string solution_code;
    std::vector<std::pair<std::string, float>> word_vector;
    double success_timestamp;
    int api_calls_saved = 0;
};

std::string get_env(const std::string& key) {
    const char* val = std::getenv(key.c_str());
    return val ? std::string(val) : "";
}

std::string md5_hash(const std::string& input) {
    unsigned char digest[MD5_DIGEST_LENGTH];
    MD5(reinterpret_cast<const unsigned char*>(input.c_str()), input.length(), digest);
    char hash_str[33];
    for(int i = 0; i < 16; i++) {
        sprintf(&hash_str[i*2], "%02x", digest[i]);
    }
    return std::string(hash_str).substr(0, 8);
}

std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

double get_time() {
    auto now = std::chrono::system_clock::now();
    return std::chrono::duration<double>(now.time_since_epoch()).count();
}

static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string lower;
    lower.resize(text.size());
    std::transform(text.begin(), text.end(), lower.begin(), ::tolower);
    std::regex word_regex(R"(\b[a-z]{3,}\b)");
    auto words_begin = std::sregex_iterator(lower.begin(), lower.end(), word_regex);
    auto words_end = std::sregex_iterator();
    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        tokens.push_back((*i).str());
    }
    return tokens;
}

std::vector<std::pair<std::string, float>> compute_tf_vector(const std::string& text) {
    auto tokens = tokenize(text);
    std::map<std::string, int> freq;
    for (const auto& t : tokens) freq[t]++;
    std::vector<std::pair<std::string, float>> vec;
    float max_freq = 0;
    for (const auto& [k, v] : freq) max_freq = std::max(max_freq, (float)v);
    for (const auto& [k, v] : freq) {
        vec.push_back({k, v / max_freq});
    }
    return vec;
}

float cosine_similarity(const std::vector<std::pair<std::string, float>>& v1,
                       const std::vector<std::pair<std::string, float>>& v2) {
    std::map<std::string, float> m1(v1.begin(), v1.end());
    std::map<std::string, float> m2(v2.begin(), v2.end());
    float dot = 0, norm1 = 0, norm2 = 0;
    for (const auto& [k, val] : m1) {
        norm1 += val * val;
        auto it = m2.find(k);
        if (it != m2.end()) dot += val * it->second;
    }
    for (const auto& [k, val] : m2) norm2 += val * val;
    if (norm1 == 0 || norm2 == 0) return 0;
    return dot / (std::sqrt(norm1) * std::sqrt(norm2));
}

class GroqClient {
private:
    std::string api_key;
    std::string base_url = "https://api.groq.com/openai/v1/chat/completions";
    std::string shorten_system_prompt(const std::string& original) const {
        std::string shortened = original;
        std::regex rules_section(R"(CRITICAL RULES FOR DATA TASKS:[\s\S]*?If execution fails -> declare outcome: continue and propose fixes)");
        shortened = std::regex_replace(shortened, rules_section, "CRITICAL: Check return code, read output file before success, never lie.");
        return shortened;
    }
public:
    GroqClient(const std::string& key) : api_key(key) {
        if(api_key.empty()) throw std::runtime_error("GROQ_API_KEY not set.");
    }
    struct ChatResponse { std::string content; bool success; std::string error; };
    ChatResponse chat(const std::string& model, const std::vector<std::pair<std::string, std::string>>& messages,
                      double temperature = 0.0, int max_tokens = 1024, bool stream = false) {
        const int MAX_RETRIES = 3;
        std::string system_prompt = messages.empty() ? "" : messages[0].second;
        for (int attempt = 0; attempt < MAX_RETRIES; ++attempt) {
            CURL* curl = curl_easy_init();
            if (!curl) return {"", false, "CURL init failed"};
            std::string response_string;
            ChatResponse result{"", false, ""};
            json payload;
            payload["model"] = model;
            payload["temperature"] = temperature;
            payload["max_tokens"] = max_tokens;
            payload["stream"] = stream;
            json msgs = json::array();
            std::string effective_system = (attempt > 0) ? shorten_system_prompt(system_prompt) : system_prompt;
            for (const auto& msg : messages) {
                json m;
                m["role"] = msg.first;
                m["content"] = (msg.first == "system" && attempt > 0) ? effective_system : msg.second;
                msgs.push_back(m);
            }
            payload["messages"] = msgs;
            std::string json_str = payload.dump();
            struct curl_slist* headers = nullptr;
            headers = curl_slist_append(headers, "Content-Type: application/json");
            headers = curl_slist_append(headers, ("Authorization: Bearer " + api_key).c_str());
            curl_easy_setopt(curl, CURLOPT_URL, base_url.c_str());
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_str.c_str());
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);
            curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L);
            CURLcode res = curl_easy_perform(curl);
            if (res != CURLE_OK) {
                result.error = std::string("CURL error: ") + curl_easy_strerror(res);
            } else {
                try {
                    json response = json::parse(response_string);
                    if (response.contains("choices") && !response["choices"].empty()) {
                        result.content = response["choices"][0]["message"]["content"];
                        result.success = true;
                        curl_slist_free_all(headers);
                        curl_easy_cleanup(curl);
                        return result;
                    } else if (response.contains("error")) {
                        result.error = response["error"]["message"];
                    }
                } catch (...) {
                    result.error = "JSON parse error";
                }
            }
            curl_slist_free_all(headers);
            curl_easy_cleanup(curl);
            if (result.error.find("Rate limit") != std::string::npos || 
                result.error.find("429") != std::string::npos || 
                result.error.find("rate_limit") != std::string::npos) {
                int backoff_ms = (1 << attempt) * 10000 + (rand() % 5000);
                std::cout << "[RATE LIMIT] Attempt " << (attempt+1) << "/" << MAX_RETRIES 
                          << " - backing off " << backoff_ms << "ms\n";
                std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
                continue;
            }
            return result;
        }
        return {"", false, "Rate limit retry exhausted"};
    }
};

class SemanticMemoryManager {
private:
    AgentConfig config;
    std::vector<SemanticMemoryEntry> memory_store;
    fs::path memory_file;
public:
    SemanticMemoryManager(const AgentConfig& cfg) : config(cfg) {
        fs::create_directories(config.memory_db_path);
        memory_file = fs::path(config.memory_db_path) / "semantic_memory.json";
        load_memory();
    }
    void load_memory() {
        if(fs::exists(memory_file)) {
            try {
                std::ifstream f(memory_file);
                json data;
                f >> data;
                for(const auto& item : data) {
                    SemanticMemoryEntry entry;
                    entry.task_hash = item["task_hash"];
                    entry.task_text = item["task_text"];
                    entry.solution_code = item["solution_code"];
                    entry.success_timestamp = item["timestamp"];
                    entry.api_calls_saved = item.value("api_calls_saved", 0);
                    for(const auto& [k, v] : item["vector"].items()) {
                        entry.word_vector.push_back({k, v.get<float>()});
                    }
                    memory_store.push_back(entry);
                }
                std::cout << "[SemanticMemory] Loaded " << memory_store.size() << " entries\n";
            } catch(...) {}
        }
    }
    void save_memory() {
        try {
            json data = json::array();
            for(const auto& entry : memory_store) {
                json j;
                j["task_hash"] = entry.task_hash;
                j["task_text"] = entry.task_text.substr(0, 200);
                j["solution_code"] = entry.solution_code;
                j["timestamp"] = entry.success_timestamp;
                j["api_calls_saved"] = entry.api_calls_saved;
                json vec;
                for(const auto& [k, v] : entry.word_vector) vec[k] = v;
                j["vector"] = vec;
                data.push_back(j);
            }
            std::ofstream f(memory_file);
            f << data.dump(2);
        } catch(...) {}
    }
    std::optional<std::string> retrieve_similar(const std::string& query, float threshold = 0.75f) {
        if(memory_store.empty()) return std::nullopt;
        auto query_vec = compute_tf_vector(query);
        std::pair<float, SemanticMemoryEntry*> best_match = {0, nullptr};
        for(auto& entry : memory_store) {
            float sim = cosine_similarity(query_vec, entry.word_vector);
            if(sim > best_match.first) best_match = {sim, &entry};
        }
        if(best_match.first > threshold) {
            best_match.second->api_calls_saved++;
            save_memory();
            std::cout << "[Semantic Cache] Hit! Similarity: " << best_match.first
                      << " (saved " << best_match.second->api_calls_saved << " calls)\n";
            return best_match.second->solution_code;
        }
        return std::nullopt;
    }
	std::vector<std::pair<float, std::string>> retrieve_top_k(const std::string& query, int k = 2, float min_threshold = 0.60f) {
	    if (memory_store.empty()) return {};
	    auto query_vec = compute_tf_vector(query);
	    std::vector<std::pair<float, SemanticMemoryEntry*>> scored;
	    for (auto& entry : memory_store) {
	        float sim = cosine_similarity(query_vec, entry.word_vector);
	        if (sim >= min_threshold) scored.emplace_back(sim, &entry);
	    }
	    std::sort(scored.rbegin(), scored.rend());
	    std::vector<std::pair<float, std::string>> result;
	    for (int i = 0; i < std::min(k, (int)scored.size()); ++i) {
	        result.emplace_back(scored[i].first, scored[i].second->solution_code);
	        scored[i].second->api_calls_saved++;
	    }
	    if (!result.empty()) save_memory();
	    return result;
	}
    void store(const std::string& task, const std::string& solution_code) {
        SemanticMemoryEntry entry;
        entry.task_hash = md5_hash(task);
        entry.task_text = task;
        entry.solution_code = solution_code;
        entry.word_vector = compute_tf_vector(task);
        entry.success_timestamp = get_time();
        memory_store.erase(std::remove_if(memory_store.begin(), memory_store.end(),
            [&](const auto& e) { return e.task_hash == entry.task_hash; }), memory_store.end());
        memory_store.push_back(entry);
        if(memory_store.size() > 50) {
            memory_store.erase(memory_store.begin());
        }
        save_memory();
    }
    int get_cache_hits() {
        int total = 0;
        for(const auto& e : memory_store) total += e.api_calls_saved;
        return total;
    }
};

class FileManager {
private:
    fs::path backup_dir;
public:
    std::string last_written_file;
    std::string last_written_lang;
    FileManager(const std::string& backup) : backup_dir(backup) {
        fs::create_directories(backup_dir);
    }
    std::string backup_file(const std::string& file_path) {
        fs::path path(file_path);
        if(!fs::exists(path)) return "";
        int max_version = 0;
        for(const auto& entry : fs::directory_iterator(backup_dir)) {
            std::string name = entry.path().filename().string();
            std::regex re(path.filename().string() + "_v(\\d+)\\.bak$");
            std::smatch match;
            if(std::regex_search(name, match, re)) {
                max_version = std::max(max_version, std::stoi(match[1]));
            }
        }
        fs::path backup_path = backup_dir / (path.filename().string() + "_v" + std::to_string(max_version + 1) + ".bak");
        try {
            fs::copy_file(path, backup_path, fs::copy_options::overwrite_existing);
            return backup_path.string();
        } catch(...) { return ""; }
    }
    std::string write_file(const std::string& file_path, const std::string& content) {
        std::string processed_content = content;
        size_t pos = 0;
        while((pos = processed_content.find("\\n", pos)) != std::string::npos) {
            processed_content.replace(pos, 2, "\n");
            pos++;
        }
        std::ofstream f(file_path);
        if (!f) return "Error: Could not open file for writing.";
        f << processed_content;
        f.close();
        last_written_file = file_path;
        std::string ext = fs::path(file_path).extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".cpp" || ext == ".cc" || ext == ".cxx") last_written_lang = "cpp";
        else if (ext == ".py") last_written_lang = "python";
        else last_written_lang = "unknown";
        return "File written successfully: " + file_path;
    }
    std::string read_file(const std::string& file_path) {
        try {
            std::ifstream f(file_path);
            std::stringstream buffer;
            buffer << f.rdbuf();
            return buffer.str();
        } catch(const std::exception& e) {
            return "Error reading file: " + std::string(e.what());
        }
    }
    std::string undo_last_change(const std::string& file_path) {
        fs::path path(file_path);
        std::vector<std::pair<int, fs::path>> versions;
        for(const auto& entry : fs::directory_iterator(backup_dir)) {
            std::string name = entry.path().filename().string();
            std::regex re(path.filename().string() + "_v(\\d+)\\.bak$");
            std::smatch match;
            if(std::regex_search(name, match, re)) {
                versions.push_back({std::stoi(match[1]), entry.path()});
            }
        }
        if(versions.empty()) return "No backup found.";
        auto latest = *std::max_element(versions.begin(), versions.end(),
            [](auto& a, auto& b) { return a.first < b.first; });
        try {
            fs::copy_file(latest.second, path, fs::copy_options::overwrite_existing);
            fs::remove(latest.second);
            return "Reverted " + file_path + " to version " + std::to_string(latest.first) + ".";
        } catch(const std::exception& e) {
            return "Error reverting file: " + std::string(e.what());
        }
    }
};

class CodeExecutor {
private:
    AgentConfig config;
    fs::path log_path = "./execution_traceback.log";
    std::string get_executable_name(const std::string& cpp_file) {
        fs::path p(cpp_file);
        std::string base = p.stem().string();
        return "/tmp/" + base + "_" + std::to_string(getpid()) + ".out";
    }
public:
    CodeExecutor(const AgentConfig& cfg) : config(cfg) {}
    std::string execute_with_timeout(const std::string& command, int timeout_sec) {
        std::array<char, 128> buffer;
        std::string stdout_result, stderr_result;
        int return_code = -1;
        std::string tmp_out = "/tmp/agent_out_" + std::to_string(getpid()) + ".txt";
        std::string tmp_err = "/tmp/agent_err_" + std::to_string(getpid()) + ".txt";
        std::string cmd = command + " > " + tmp_out + " 2> " + tmp_err;
        auto start = std::chrono::steady_clock::now();
        FILE* pipe = popen((cmd + "; echo $?").c_str(), "r");
        if(!pipe) return "Execution failed: popen failed";
        bool timed_out = false;
        while(true) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
            if(elapsed >= timeout_sec) {
                timed_out = true;
                pclose(pipe);
                break;
            }
            if(fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
                std::string line(buffer.data());
                try {
                    return_code = std::stoi(line);
                } catch(...) {}
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        std::ifstream out_f(tmp_out);
        std::ifstream err_f(tmp_err);
        stdout_result = std::string((std::istreambuf_iterator<char>(out_f)), std::istreambuf_iterator<char>());
        stderr_result = std::string((std::istreambuf_iterator<char>(err_f)), std::istreambuf_iterator<char>());
        std::remove(tmp_out.c_str());
        std::remove(tmp_err.c_str());
        if(timed_out) return_code = -9;
        std::ofstream log(log_path, std::ios::app);
        log << "=== " << get_timestamp() << " (rc=" << return_code << ") ===\n";
        log << stderr_result << "\n";
        log << std::string(60, '=') << "\n\n";
        return "Return code: " + std::to_string(return_code) + "\nSTDOUT:\n" +
               (stdout_result.empty() ? "(none)" : stdout_result) + "\nSTDERR:\n" +
               (stderr_result.empty() ? "(none)" : stderr_result);
    }
    std::string compile_cpp(const std::string& source_file, const std::string& output_file = "") {
        std::string exe_path = output_file.empty() ? get_executable_name(source_file) : output_file;
        std::string compile_cmd = "g++ -std=c++17 -O2 -o " + exe_path + " " + source_file;
        std::string result = execute_with_timeout(compile_cmd, config.max_code_timeout);
        if(result.find("Return code: 0") == std::string::npos) {
            return "COMPILATION_FAILED:" + result;
        }
        return exe_path;
    }
    std::string validate_cpp_syntax(const std::string& source_file) {
        std::string cmd = "g++ -std=c++17 -fsyntax-only " + source_file;
        std::string result = execute_with_timeout(cmd, 10);
        if(result.find("Return code: 0") != std::string::npos) {
            return "Syntax validation PASSED.";
        } else {
            return "Syntax validation FAILED: " + result;
        }
    }
    std::string execute_file(const std::string& file_path) {
        fs::path p(file_path);
        std::string ext = p.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if(ext == ".cpp" || ext == ".cc" || ext == ".cxx") {
            std::string exe_path = compile_cpp(file_path);
            if(exe_path.find("COMPILATION_FAILED:") == 0) {
                return exe_path.substr(19);
            }
            std::string run_result = execute_with_timeout(exe_path, config.max_execution_timeout);
            std::remove(exe_path.c_str());
            return run_result;
        } else if(ext == ".py") {
            return execute_with_timeout("python3 " + file_path, config.max_execution_timeout);
        } else {
            return execute_with_timeout(file_path, config.max_execution_timeout);
        }
    }
    std::string execute_code(const std::string& code, const std::string& language = "auto") {
        std::string lang = language;
        if(lang == "auto") {
            if(code.find("#include") != std::string::npos ||
               code.find("std::") != std::string::npos ||
               code.find("int main(") != std::string::npos) {
                lang = "cpp";
            } else {
                lang = "python";
            }
        }
        if(lang == "cpp") {
            std::string tmp_file = "/tmp/agent_code_" + std::to_string(getpid()) + ".cpp";
            std::ofstream f(tmp_file);
            f << code;
            f.close();
            std::string result = execute_file(tmp_file);
            std::remove(tmp_file.c_str());
            return result;
        } else {
            std::string syntax_check = validate_syntax(code, "python");
            if(syntax_check.find("FAILED") != std::string::npos) {
                return syntax_check + "\n(Execution skipped - fix syntax first)";
            }
            std::string tmp_file = "/tmp/agent_code_" + std::to_string(getpid()) + ".py";
            std::ofstream f(tmp_file);
            f << code;
            f.close();
            std::string result = execute_with_timeout("python3 " + tmp_file, config.max_code_timeout);
            std::remove(tmp_file.c_str());
            return result;
        }
    }
    std::string install_package(const std::string& package) {
        if(package.find("lib") == 0 || package.find("g++") == 0 || package.find("clang") == 0) {
            return execute_with_timeout("apt-get install -y " + package, config.max_install_timeout);
        }
        return execute_with_timeout("pip3 install " + package, config.max_install_timeout);
    }
    std::string validate_syntax(const std::string& code, const std::string& language = "auto") {
        std::string lang = language;
        if(lang == "auto") {
            if(code.find("#include") != std::string::npos || code.find("int main(") != std::string::npos) {
                lang = "cpp";
            } else {
                lang = "python";
            }
        }
        if(lang == "cpp") {
            std::string tmp_file = "/tmp/agent_syntax_" + std::to_string(getpid()) + ".cpp";
            std::ofstream f(tmp_file);
            f << code;
            f.close();
            std::string result = validate_cpp_syntax(tmp_file);
            std::remove(tmp_file.c_str());
            return result;
        } else {
            std::string tmp_file = "/tmp/agent_syntax_" + std::to_string(getpid()) + ".py";
            std::ofstream f(tmp_file);
            f << code;
            f.close();
            std::string result = execute_with_timeout("python3 -m py_compile " + tmp_file, 10);
            std::remove(tmp_file.c_str());
            if(result.find("Return code: 0") != std::string::npos) {
                return "Syntax validation PASSED.";
            } else {
                return "Syntax validation FAILED: " + result;
            }
        }
    }
};

class ToolCache {
private:
    std::unordered_map<std::string, std::string> cache;
public:
    std::string get_key(const std::string& tool_name, const std::map<std::string, std::string>& args) {
        std::string key_data = tool_name + ":";
        for(const auto& [k, v] : args) key_data += k + "=" + v + ";";
        return md5_hash(key_data);
    }
    std::string get(const std::string& key) {
        auto it = cache.find(key);
        return it != cache.end() ? it->second : "";
    }
    void set(const std::string& key, const std::string& value) {
        cache[key] = value;
    }
    void clear() { cache.clear(); }
};

class ReflectionEngine {
private:
    std::map<std::string, std::map<std::string, int>> patterns;
    std::vector<AnalysisResult> failure_log;
public:
    AnalysisResult analyze_session(const std::string& task, bool success, const std::string& observations, int turn_count) {
        AnalysisResult analysis;
        analysis.timestamp = get_time();
        analysis.task_hash = md5_hash(task);
        analysis.success = success;
        analysis.turn_count = turn_count;
        if(!success) {
            std::string obs_lower = observations;
            std::transform(obs_lower.begin(), obs_lower.end(), obs_lower.begin(), ::tolower);
            if(obs_lower.find("timeout") != std::string::npos || turn_count >= 9) {
                analysis.error_type = "timeout";
                analysis.bottleneck = "inefficient_algorithm_or_infinite_loop";
                analysis.suggested_capability = "complexity_analyzer";
            } else if(obs_lower.find("return code: 1") != std::string::npos) {
                analysis.error_type = "execution_error";
                if(obs_lower.find("error:") != std::string::npos) {
                    if(obs_lower.find("cpp") != std::string::npos || obs_lower.find("g++") != std::string::npos) {
                        analysis.suggested_capability = "cpp_compiler_helper";
                    } else if(obs_lower.find("python") != std::string::npos || obs_lower.find("syntaxerror") != std::string::npos) {
                        analysis.suggested_capability = "python_syntax_helper";
                    } else if(task.find("sqlite") != std::string::npos || task.find("SQLite") != std::string::npos) {
                        analysis.suggested_capability = "sql_debugger";
                    } else if(obs_lower.find("import") != std::string::npos) {
                        analysis.suggested_capability = "dependency_resolver";
                    }
                }
            } else if(obs_lower.find("compilation") != std::string::npos || obs_lower.find("g++") != std::string::npos) {
                analysis.error_type = "compilation_error";
                analysis.suggested_capability = "cpp_compiler_helper";
            } else if(obs_lower.find("no such file") != std::string::npos) {
                analysis.error_type = "missing_file";
                analysis.suggested_capability = "path_validator";
            }
            patterns[analysis.error_type]["count"]++;
            failure_log.push_back(analysis);
        }
        return analysis;
    }
    json get_stats() {
        json stats;
        stats["total_failures"] = failure_log.size();
        stats["pattern_counts"] = patterns;
        return stats;
    }
};

class CodingAgent {
public:
    AgentConfig config;
    EvolutionConfig evolution_config;
    std::unique_ptr<GroqClient> client;
    FileManager file_manager;
    CodeExecutor executor;
    SemanticMemoryManager semantic_memory;
    ToolCache cache;
    ReflectionEngine reflection;
    std::vector<std::string> execution_history;
    std::string last_observations;
    bool session_has_successful_execution = false;
    int api_calls_avoided = 0;
    CodingAgent(const AgentConfig& cfg, const EvolutionConfig& evo);
    std::string list_directory_files();
    std::string read_file(const std::string& file_path);
    std::string write_file(const std::string& file_path, const std::string& content);
    std::string make_directory(const std::string& dir_path);
    std::string undo_last_change(const std::string& file_path);
    std::string execute_file(const std::string& file_path);
    std::string execute_python_file(const std::string& file_path) { return execute_file(file_path); }
    std::string run_code(const std::string& code, const std::string& language = "auto");
    std::string install_package(const std::string& package);
    std::string search_in_files(const std::string& pattern, const std::vector<std::string>& include_extensions = {});
    std::string get_environment_info();
    std::string clear_cache();
    std::string run(const std::string& task_description, int max_turns = 10);
private:
    std::string get_system_prompt();
    std::string parse_and_run_actions(const std::string& text);
    bool check_outcome(const std::string& text);
    std::string auto_verify_execution(const std::string& observations);
    std::vector<std::string> prune_context(const std::vector<std::string>& context);
    void reflect(const std::string& task, bool success, const std::string& observations, int turn_count);
    std::string detect_language(const std::string& file_path);
};

CodingAgent::CodingAgent(const AgentConfig& cfg, const EvolutionConfig& evo)
    : config(cfg), evolution_config(evo), file_manager(cfg.backup_dir), executor(cfg), semantic_memory(cfg) {
    std::string api_key = get_env("GROQ_API_KEY");
    if(api_key.empty()) throw std::runtime_error("GROQ_API_KEY environment variable is not set.");
    client = std::make_unique<GroqClient>(api_key);
    std::cout << "[Agent] Initialized with Groq (model: " << config.model_id << "). Evolution: " << (evolution_config.enabled ? "ON" : "OFF") << "\n";
    std::cout << "[Agent] Semantic Cache: ENABLED (threshold: " << config.semantic_similarity_threshold << ")\n";
    std::cout << "[Agent] Multi-language support: C++ & Python\n";
    if(evolution_config.enabled) {
        std::cout << " - Auto-reflect: " << (evolution_config.auto_reflect ? "true" : "false") << "\n";
        std::cout << " - Context window: " << evolution_config.max_context_turns << " turns\n";
    }
}

std::string CodingAgent::detect_language(const std::string& file_path) {
    fs::path p(file_path);
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if(ext == ".cpp" || ext == ".cc" || ext == ".cxx" || ext == ".hpp") return "cpp";
    if(ext == ".py") return "python";
    return "unknown";
}

std::string CodingAgent::list_directory_files() {
    std::map<std::string, std::string> args;
    std::string cache_key = cache.get_key("list_directory_files", args);
    std::string cached = cache.get(cache_key);
    if(!cached.empty()) return cached;
    std::vector<std::string> files;
    std::unordered_set<std::string> ignored = {".git", "venv", "__pycache__", "node_modules", ".venv", config.backup_dir, ".agent_backups"};
    for(const auto& entry : fs::recursive_directory_iterator(".", fs::directory_options::skip_permission_denied)) {
        if(entry.is_regular_file()) {
            std::string rel_path = fs::relative(entry.path(), ".").string();
            bool skip = false;
            for(const auto& ign : ignored) {
                if(rel_path.find(ign) != std::string::npos) { skip = true; break; }
            }
            if(!skip) files.push_back(rel_path);
        }
    }
    std::string result;
    for(const auto& f : files) result += f + "\n";
    cache.set(cache_key, result);
    return result;
}

std::string CodingAgent::read_file(const std::string& file_path) {
    std::map<std::string, std::string> args{{"file_path", file_path}};
    std::string cache_key = cache.get_key("read_file", args);
    std::string cached = cache.get(cache_key);
    if(!cached.empty()) return cached;
    std::string result = file_manager.read_file(file_path);
    cache.set(cache_key, result);
    return result;
}

std::string CodingAgent::write_file(const std::string& file_path, const std::string& content) {
    std::string lang = detect_language(file_path);
    if(lang == "cpp" || lang == "python") {
        std::string validation = executor.validate_syntax(content, lang);
        if(validation.find("FAILED") != std::string::npos) {
            std::cout << "[Pre-write Guard] Syntax error in " << file_path << ". Attempting auto-correction...\n";
            std::string fixed = content;
            bool success = false;
            for (int attempt = 0; attempt < 2 && !success; ++attempt) {
                std::string fix_prompt = "Fix this " + lang + " syntax error. The code must be valid " +
                                       (lang == "cpp" ? "C++17" : "Python 3") +
                                       ". Return ONLY the corrected code, no explanations, no markdown fences.\n\n"
                                       "Validation Error: " + validation + "\n\nCode to fix:\n" + fixed;
                std::vector<std::pair<std::string, std::string>> fix_messages = {
                    {"system", "You are a syntax correction engine. Output only valid code, no markdown."},
                    {"user", fix_prompt}
                };
                auto fix_response = client->chat(config.model_id, fix_messages, 0.1, 2048);
                if (fix_response.success && !fix_response.content.empty()) {
                    std::string corrected = fix_response.content;
                    corrected = std::regex_replace(corrected, std::regex("^```cpp\\n*"), "");
                    corrected = std::regex_replace(corrected, std::regex("^```python\\n*"), "");
                    corrected = std::regex_replace(corrected, std::regex("^```\\n*"), "");
                    corrected = std::regex_replace(corrected, std::regex("\\n*```$"), "");
                    // Extra fallback for unterminated string in Python
                    if (lang == "python" && corrected.find("f.write(\"") != std::string::npos && corrected.find("\")") == std::string::npos) {
                        size_t pos = corrected.find("f.write(\"");
                        if (pos != std::string::npos) {
                            size_t end = corrected.find("\n", pos);
                            if (end != std::string::npos && corrected[end-1] != '"') {
                                corrected.insert(end, "\")");
                            }
                        }
                    }
                    std::string revalidation = executor.validate_syntax(corrected, lang);
                    if (revalidation.find("PASSED") != std::string::npos) {
                        std::cout << "[Pre-write Guard] Auto-correction successful.\n";
                        fixed = corrected;
                        success = true;
                        break;
                    } else {
                        std::cout << "[Pre-write Guard] Auto-correction attempt " << attempt+1 << " failed validation.\n";
                        validation = revalidation;
                    }
                }
            }
            if (success) {
                std::string result = file_manager.write_file(file_path, fixed);
                cache.clear();
                return result + " [Syntax auto-corrected]";
            } else {
                return "CRITICAL ERROR: Syntax validation failed for " + file_path +
                       " and auto-correction was unsuccessful. Original error: " + validation;
            }
        }
    }
    std::string result = file_manager.write_file(file_path, content);
    cache.clear();
    return result;
}

std::string CodingAgent::make_directory(const std::string& dir_path) {
    try {
        fs::create_directories(dir_path);
        cache.clear();
        return "Successfully created directory " + dir_path + ".";
    } catch(const std::exception& e) {
        return "Error creating directory " + dir_path + ": " + e.what();
    }
}

std::string CodingAgent::undo_last_change(const std::string& file_path) {
    std::string result = file_manager.undo_last_change(file_path);
    cache.clear();
    return result;
}

std::string CodingAgent::execute_file(const std::string& file_path) {
    std::string result = executor.execute_file(file_path);
    cache.clear();
    if(result.find("Return code: 0") != std::string::npos) {
        session_has_successful_execution = true;
    }
    return result;
}

std::string CodingAgent::run_code(const std::string& code, const std::string& language) {
    return executor.execute_code(code, language);
}

std::string CodingAgent::install_package(const std::string& package) {
    return executor.install_package(package);
}

std::string CodingAgent::search_in_files(const std::string& pattern, const std::vector<std::string>& include_extensions) {
    std::map<std::string, std::string> args{{"pattern", pattern}};
    std::string ext_str;
    for(const auto& e : include_extensions) ext_str += e + ",";
    args["include_extensions"] = ext_str;
    std::string cache_key = cache.get_key("search_in_files", args);
    std::string cached = cache.get(cache_key);
    if(!cached.empty()) return cached;
    std::string result;
    try {
        std::string cmd = "grep -r -n \"" + pattern + "\" .";
        if(!include_extensions.empty()) {
            for(const auto& ext : include_extensions) {
                cmd += " --include=*." + ext;
            }
        }
        cmd += " --exclude-dir=.git --exclude-dir=venv --exclude-dir=__pycache__ --exclude-dir=node_modules --exclude-dir=.venv --exclude-dir=" + config.backup_dir;
        FILE* pipe = popen(cmd.c_str(), "r");
        if(!pipe) result = "Search failed: popen failed";
        else {
            char buffer[256];
            while(fgets(buffer, sizeof(buffer), pipe) != nullptr) result += buffer;
            int rc = pclose(pipe);
            if(rc == 1 && result.empty()) result = "No matches found.";
            else if(rc != 0 && rc != 1) result = "Search failed with code: " + std::to_string(rc);
        }
    } catch(const std::exception& e) {
        result = "Error during search: " + std::string(e.what());
    }
    cache.set(cache_key, result);
    return result.empty() ? "No matches found." : result;
}

std::string CodingAgent::get_environment_info() {
    std::map<std::string, std::string> args;
    std::string cache_key = cache.get_key("get_environment_info", args);
    std::string cached = cache.get(cache_key);
    if(!cached.empty()) return cached;
    std::string result = "=== Python Packages ===\n";
    FILE* pipe = popen("pip3 list 2>/dev/null || echo 'pip3 not available'", "r");
    if(pipe) {
        char buffer[128];
        while(fgets(buffer, sizeof(buffer), pipe) != nullptr) result += buffer;
        pclose(pipe);
    }
    result += "\n=== C++ Compiler ===\n";
    pipe = popen("g++ --version 2>/dev/null || echo 'g++ not available'", "r");
    if(pipe) {
        char buffer[256];
        while(fgets(buffer, sizeof(buffer), pipe) != nullptr) result += buffer;
        pclose(pipe);
    }
    cache.set(cache_key, result);
    return result;
}

std::string CodingAgent::clear_cache() {
    cache.clear();
    return "Cache cleared.";
}

std::string CodingAgent::get_system_prompt() {
    return R"x(You are an expert coding agent with full filesystem access supporting both C++ and Python.
Work step-by-step and choose the appropriate language based on the task requirements.
ABSOLUTE RULES - VIOLATING THESE CAUSES IMMEDIATE FAILURE:
- You MUST check the return code of execute_file (formerly execute_python_file)
- If return code != 0 -> you MUST output outcome: continue
- You MUST read the target output file and confirm it exists AND contains correct-looking data BEFORE declaring success
- For C++ tasks: code must compile with g++ -std=c++17 without errors before execution
- Never lie about success when observations show clear failure
LANGUAGE SELECTION GUIDE:
- Use C++ for: performance-critical algorithms, systems programming, Monte Carlo simulations, numerical computing
- Use Python for: data processing, file I/O operations, rapid prototyping, SQLite tasks
- Default to C++ if the task mentions "C++ program", "compile", or performance requirements
You have these tools:
- list_directory_files()
- read_file(file_path: str)
- write_file(file_path: str, content: str)  - Auto-validates syntax for .cpp and .py
- make_directory(dir_path: str)
- execute_file(file_path: str)  - Auto-detects language from extension (.cpp -> compile+run, .py -> interpret)
- execute_python_file(file_path: str)  - Legacy alias for execute_file
- undo_last_change(file_path: str)
- run_code(code: str, language: str = "auto")  - language can be "cpp", "python", or "auto"
- install_package(package: str)  - For C++ libs (lib*) or Python packages
- search_in_files(pattern: str, include_extensions: list[str] = None)
- get_environment_info()
- clear_cache()
CRITICAL RULES FOR C++ TASKS:
- ALWAYS ensure #include <bits/stdc++.h> or specific headers are present
- Use std::cout for output, std::cerr for errors
- For compilation errors, read the error message carefully and fix the code
- Use -std=c++17 -O2 flags for compilation (handled automatically)
- ALWAYS use "tool": "tool_name" in actions JSON. Never use "action".
- For multiline code in JSON, ALWAYS escape as \n. NEVER use raw newlines or """ inside the JSON.
CRITICAL RULES FOR DATA TASKS:
- ALWAYS use "tool": "tool_name" in actions JSON. Never use "action".
- For multiline code in JSON, ALWAYS escape as \n. NEVER use raw newlines or """ inside the JSON.
- When the task explicitly says "Create XXX.cpp" or "Create XXX.py", use exactly that filename with write_file.
- For cleaning dirty CSV/JSON, ALWAYS use try/except around int()/float().
- Only declare outcome: success after you have seen correct numeric results.
STRICT OUTPUT FORMAT:
1. Step-by-step reasoning in plain text.
2. One or more ```actions``` JSON blocks.
3. Exactly one ```yaml``` outcome block at the very end.
Actions format example for C++:
```actions
[
  {
    "tool": "write_file",
    "args": {
      "file_path": "program.cpp",
      "content": "#include <iostream>\nint main() { std::cout << \"Hello\" << std::endl; return 0; }"
    }
  },
  {
    "tool": "execute_file",
    "args": {
      "file_path": "program.cpp"
    }
  }
]
```
Outcome block (must be last):
```yaml
outcome: success
```
or
```yaml
outcome: continue
```)x";
}

static std::string repair_json_newlines(const std::string& json_str) {
    std::string result;
    bool in_string = false;
    bool escape = false;
    for (size_t i = 0; i < json_str.size(); ++i) {
        char c = json_str[i];
        if (escape) {
            result += c;
            escape = false;
            continue;
        }
        if (c == '\\') {
            escape = true;
            result += c;
            continue;
        }
        if (c == '"') {
            in_string = !in_string;
            result += c;
            continue;
        }
        if (in_string && c == '\n') {
            result += "\\n";
            continue;
        }
        result += c;
    }
    return result;
}

std::string CodingAgent::parse_and_run_actions(const std::string& text) {
    std::regex pattern(R"(```actions\s*([\s\S]*?)```)");
    std::sregex_iterator iter(text.begin(), text.end(), pattern);
    std::sregex_iterator end_iter;
    std::vector<std::string> matches;
    for (; iter != end_iter; ++iter)
        matches.push_back((*iter)[1].str());
    if (matches.empty())
        return "Observation: No actions block found.";
    std::vector<std::string> all_observations;
    for (const auto& raw_block : matches) {
        std::string normalized = raw_block;
        normalized = std::regex_replace(normalized, std::regex(R"("action")"), "\"tool\"");
        std::string repaired = repair_json_newlines(normalized);
        json actions;
        try {
            actions = json::parse(repaired);
        } catch (const json::parse_error& e) {
            return "Observation: JSON parse error in actions block: " + std::string(e.what());
        }
        if (!actions.is_array())
            actions = json::array({actions});
        std::vector<std::string> block_results;
        for (auto& action : actions) {
            if (action.contains("action") && !action.contains("tool")) {
                action["tool"] = action["action"];
                action.erase("action");
            }
            if (!action.contains("tool") || !action.contains("args"))
                continue;
            std::string tool_name = action["tool"];
            json args = action["args"];
            std::string result;
            if (args.contains("content") && args["content"].is_string()) {
                std::string content = args["content"];
                size_t pos = 0;
                while ((pos = content.find("\\n", pos)) != std::string::npos) {
                    content.replace(pos, 2, "\n");
                    pos++;
                }
                args["content"] = content;
            }
            if (tool_name == "write_file") {
                result = write_file(args["file_path"], args["content"]);
            } else if (tool_name == "read_file") {
                result = read_file(args["file_path"]);
            } else if (tool_name == "list_directory_files") {
                result = list_directory_files();
            } else if (tool_name == "make_directory") {
                result = make_directory(args["dir_path"]);
            } else if (tool_name == "execute_file" || tool_name == "execute_python_file") {
                result = execute_file(args["file_path"]);
            } else if (tool_name == "undo_last_change") {
                result = undo_last_change(args["file_path"]);
            } else if (tool_name == "run_python_code" || tool_name == "run_code") {
                std::string lang = args.value("language", "auto");
                if (args.contains("code"))
                    result = run_code(args["code"], lang);
                else if (args.contains("python_code"))
                    result = run_code(args["python_code"], "python");
            } else if (tool_name == "install_package") {
                result = install_package(args["package"]);
            } else if (tool_name == "search_in_files") {
                std::vector<std::string> exts;
                if (args.contains("include_extensions")) {
                    for (const auto& e : args["include_extensions"])
                        exts.push_back(e.get<std::string>());
                }
                result = search_in_files(args["pattern"], exts);
            } else if (tool_name == "get_environment_info") {
                result = get_environment_info();
            } else if (tool_name == "clear_cache") {
                result = clear_cache();
            } else {
                result = "Observation: Unknown tool '" + tool_name + "'";
            }
            block_results.push_back("Tool " + tool_name + " Result: " + result);
        }
        std::string block_str;
        for (const auto& r : block_results)
            block_str += r + "\n";
        all_observations.push_back(block_str);
    }
    std::string final_result;
    for (const auto& obs : all_observations)
        final_result += obs + "\n\n";
    return final_result;
}

bool CodingAgent::check_outcome(const std::string& text) {
    std::regex success_re(R"(```yaml\s*outcome:\s*success\s*```)", std::regex::icase);
    bool declared = std::regex_search(text, success_re);
    if (!declared) return false;
    bool has_valid_execution = session_has_successful_execution ||
        (!last_observations.empty() && last_observations.find("Return code: 0") != std::string::npos);
    if (!has_valid_execution) {
        std::cout << "[Guard] Blocking premature success: no successful execution in session\n";
        return false;
    }
    return true;
}

std::string CodingAgent::auto_verify_execution(const std::string& observations) {
    if(observations.find("Tool execute_file Result:") != std::string::npos ||
       observations.find("Tool execute_python_file Result:") != std::string::npos) {
        return observations;
    }
    std::string extra;
    if(!file_manager.last_written_file.empty()) {
        std::cout << "Auto-verification: Re-running last written file → " << file_manager.last_written_file
                  << " (" << file_manager.last_written_lang << ")\n";
        std::string exec_result = execute_file(file_manager.last_written_file);
        extra += "\n--- Auto Execution of " + file_manager.last_written_file + " ---\n" + exec_result;
    }
    return extra.empty() ? observations : observations + extra;
}

std::vector<std::string> CodingAgent::prune_context(const std::vector<std::string>& context) {
    if((int)context.size() <= evolution_config.max_context_turns) return context;
    std::vector<std::string> pruned;
    pruned.push_back(context[0]);
    for(size_t i = context.size() - (evolution_config.max_context_turns - 1); i < context.size(); i++)
        pruned.push_back(context[i]);
    return pruned;
}

void CodingAgent::reflect(const std::string& task, bool success, const std::string& observations, int turn_count) {
    if(!evolution_config.auto_reflect) return;
    AnalysisResult analysis = reflection.analyze_session(task, success, observations, turn_count);
    if(!success && !analysis.suggested_capability.empty()) {
        std::cout << "[Reflection] Detected " << analysis.error_type
                  << " - Suggestion: " << analysis.suggested_capability << "\n";
    }
}

std::string CodingAgent::run(const std::string& task_description, int max_turns) {
    session_has_successful_execution = false;
    std::cout << "Starting task: " << task_description.substr(0, 100) << "...\n\n";
    auto cached_opt = semantic_memory.retrieve_similar(task_description, config.semantic_similarity_threshold);
    if (cached_opt.has_value()) {
        std::cout << "[Cache] Executing cached solution...\n";
        std::string observations = parse_and_run_actions(cached_opt.value());
        last_observations = observations;
        if (check_outcome(cached_opt.value() + "\n```yaml\noutcome: success\n```")) {
            std::cout << "✓ Task completed successfully via semantic cache.\n";
            api_calls_avoided++;
            return cached_opt.value() + "\n[Retrieved from semantic cache]";
        } else {
            std::cout << "[Cache] Cached solution failed, falling back to LLM...\n";
        }
    }
    auto similar = semantic_memory.retrieve_top_k(task_description, 2, 0.60f);
    std::string few_shot;
    if (!similar.empty()) {
        few_shot = "\n\n=== SIMILAR PAST SUCCESSFUL SOLUTIONS (use as inspiration) ===\n";
        for (const auto& [sim, code] : similar) {
            few_shot += "[Similarity " + std::to_string(sim) + "]\n" + code + "\n\n";
        }
        few_shot += "=== END PAST SOLUTIONS ===\n";
        std::cout << "[Semantic RAG] Injected " << similar.size() << " few-shot examples (no extra API call)\n";
    }
    std::vector<std::string> session_context;
    std::string system_prompt = get_system_prompt();
    bool final_success = false;
    std::string final_output;
    for (int turn = 0; turn < max_turns; turn++) {
        std::cout << "\n" << std::string(60, '=') << "\nTurn " << (turn + 1) << "/" << max_turns
                  << " [Cache hits: " << semantic_memory.get_cache_hits()
                  << " | RAG examples: " << (!similar.empty() ? std::to_string(similar.size()) : "0")
                  << "]\n" << std::string(60, '=') << "\n\n";

        session_context = prune_context(session_context);
        std::string history;
        for (const auto& ctx : session_context) history += ctx + "\n";
        std::string full_prompt = "Session History:\n" + history +
                                  (turn == 0 ? few_shot : "") +
                                  "\n\nTask: " + task_description +
                                  "\n\nWhat do you do next?";
        std::vector<std::pair<std::string, std::string>> messages = {
            {"system", system_prompt},
            {"user", full_prompt}
        };
        auto response = client->chat(config.model_id, messages, 0.0, 2048, false);
        if (!response.success) {
            std::cout << "API error: " << response.error << "\n";
            return "Agent failed due to API error: " + response.error;
        }
        std::string turn_text = response.content;
        std::cout << "--- Model Response ---\n" << turn_text << "\n" << std::string(60, '-') << "\n\n";
        std::string observations = parse_and_run_actions(turn_text);
        last_observations = observations;
        std::cout << "--- Observations ---\n" << observations << "\n" << std::string(60, '-') << "\n\n";
        bool success_declared = check_outcome(turn_text);
        if (success_declared) {
            observations = auto_verify_execution(observations);
            std::cout << "Auto-verified observations:\n" << observations << "\n";
        }
        if (!success_declared && observations.find("Return code: 1") != std::string::npos) {
            session_context.push_back("REMINDER: The last execution failed (return code != 0). You MUST fix the error before declaring success. Do NOT output 'outcome: success' until you see a zero return code.");
        }
        reflect(task_description, success_declared, observations, turn + 1);
        session_context.push_back("Turn " + std::to_string(turn + 1) + " Output:\n" + turn_text +
                                  "\n\nObservation:\n" + observations);
        final_output = turn_text;
        if (success_declared) {
            std::cout << "✓ Task completed successfully.\n";
            semantic_memory.store(task_description, turn_text);
            final_success = true;
            break;
        }
    }
    if (!final_success) {
        std::cout << "✗ Task failed to complete within turn limit.\n";
        return "TASK FAILED: Agent could not produce a working solution within the turn limit. Last output was:\n" + final_output;
    }
    return final_output;
}

class TestSuite {
private:
    fs::path suite_path;
    std::vector<TestCase> cases;
public:
    TestSuite(const std::string& path) : suite_path(path) {
        fs::create_directories(suite_path);
        define_default_cases();
    }
    void define_default_cases() {
        cases = {
            {"T1", "Prime Factor Optimization", "Find largest prime factor with optimization on timeout",
             R"(Create a script prime_factor.py that finds the largest prime factor of 600851475143. Start with basic trial division, but if execution exceeds 1 second, refactor to O(√n) algorithm. Print result and time taken.)",
             {"prime_factor.py"}, {{"file_exists", "prime_factor.py", "", ""}}, 4, 120},
            {"T2", "Import Restructuring", "Handle directory moves and import updates",
             R"(1. Create math_lib/operations.py with power(a,b) function. Create app.py importing it and printing 2**10. 2. Move operations.py to core/utils/, update imports, verify output still 1024.)",
             {"app.py", "core/utils/operations.py"}, {{"execution", "", "app.py", "1024"}}, 4, 120},
            {"T3", "Vending Machine OOP", "Custom exceptions and state management",
             R"(Create vending.py with VendingMachine class and InsufficientFundsError. Demonstrate: deposit $2.00, try buying $2.50 item (catch error), deposit $1 more, buy successfully. Print balance and inventory.)",
             {"vending.py"}, {{"execution", "", "vending.py", ""}}, 4, 120},
            {"T4", "Data Cleaning Pipeline", "Handle dirty CSV data with type validation",
             R"(Create raw_data.csv (Name, Age, Salary) with 6 rows including non-numeric ages/empty salaries. Create cleaner.py to filter invalid rows, calculate average salary, save to processed_data.json.)",
             {"raw_data.csv", "cleaner.py", "processed_data.json"}, {{"file_exists", "processed_data.json", "", ""}}, 6, 120},
            {"T5", "Pi Approximation", "Numerical methods and convergence",
             R"(Create pi_approx.py using Leibniz formula: π = 4 * Σ((-1)^n/(2n+1)). Iterate until |approximation - math.pi| < 10^-5. Print iterations needed.)",
             {"pi_approx.py"}, {{"execution", "", "pi_approx.py", ""}}, 4, 120},
            {"T6", "Grid BFS Pathfinding", "Graph traversal and obstacle handling",
             R"(Create grid_bfs.py with 10x10 grid, obstacles. Implement BFS from (0,0) to (9,9). Output path coordinates and step count. Handle no-path case.)",
             {"grid_bfs.py"}, {{"execution", "", "grid_bfs.py", ""}}, 4, 120},
            {"T7", "SQLite Relational Query", "Complex JOINs and subqueries",
             R"(Create SQLite database with Departments and Employees tables. Insert 3 departments, 10 employees. Query: find employees earning more than their dept average. Save to high_earners.json. Use proper foreign keys.)",
             {"high_earners.json"}, {{"file_exists", "high_earners.json", "", ""}}, 12, 120},
            {"T8", "Memoization Decorator", "Higher-order functions and performance",
             R"(Create fibonacci_memoized.py with memoize decorator and apply to recursive Fibonacci. Calculate F(50) with timing. Compare to F(30) without decorator.)",
             {"fibonacci_memoized.py"}, {{"execution", "", "fibonacci_memoized.py", ""}}, 6, 120},
            {"T9", "Monte Carlo Integration", "Probabilistic numerical methods",
             R"(Create monte_carlo.py to estimate ∫sin(x)dx from 0 to π using 1,000,000 random points. Compare estimate to analytical value 2. Print absolute error.)",
             {"monte_carlo.py"}, {{"execution", "", "monte_carlo.py", ""}}, 4, 120},
            {"T10", "Concurrent Hashing", "Parallel processing with concurrent.futures",
             R"(Create concurrent_hash.py scanning for .txt/.py files, calculate SHA-256 in parallel. Store {filename: hash} dict. Print execution time and verify count matches.)",
             {"concurrent_hash.py"}, {{"execution", "", "concurrent_hash.py", ""}}, 4, 120},
        };
    }
    std::pair<bool, std::vector<std::string>> validate_result(const TestCase& test_case, CodingAgent* agent) {
        bool passed = true;
        std::vector<std::string> details;
        for(const auto& check : test_case.validation_checks) {
            if(check.type == "file_exists") {
                if(!fs::exists(check.path)) {
                    passed = false;
                    details.push_back("Missing file: " + check.path);
                } else {
                    details.push_back("✓ Found: " + check.path);
                }
            } else if(check.type == "execution") {
                if(!check.file.empty() && fs::exists(check.file)) {
                    std::string result = agent->execute_file(check.file);
                    if(result.find("Return code: 0") == std::string::npos) {
                        passed = false;
                        details.push_back("Execution failed: " + result.substr(0, 200));
                    } else {
                        details.push_back("✓ Executed: " + check.file);
                        if(!check.expect.empty()) {
                            if(result.find(check.expect) == std::string::npos) {
                                passed = false;
                                details.push_back("Expected output '" + check.expect + "' not found in:\n" + result);
                            } else {
                                details.push_back("✓ Found expected output: " + check.expect);
                            }
                        }
                    }
                } else {
                    passed = false;
                    details.push_back("File not found for execution: " + check.file);
                }
            }
        }
        return {passed, details};
    }
    json run_evaluation(CodingAgent* agent, const std::vector<std::string>& specific_tests = {}) {
        std::cout << "\n" << std::string(60, '=') << "\nRUNNING TEST SUITE EVALUATION\n" << std::string(60, '=') << "\n";
        std::vector<TestResult> results;
        for(const auto& test_case : cases) {
            if(!specific_tests.empty() && std::find(specific_tests.begin(), specific_tests.end(), test_case.id) == specific_tests.end())
                continue;
            std::cout << "\n[" << test_case.id << "] " << test_case.name << "\n" << std::string(40, '-') << "\n";
            for(const auto& f : test_case.expected_outputs) {
                if(fs::exists(f)) fs::remove(f);
            }
            auto start = std::chrono::steady_clock::now();
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            try {
                std::string output = agent->run(test_case.task_prompt, test_case.max_turns);
                auto end = std::chrono::steady_clock::now();
                double duration = std::chrono::duration<double>(end - start).count();
                bool success = (output.find("outcome: success") != std::string::npos ||
                               output.find("✓ Task completed") != std::string::npos);
                auto [validated, details] = validate_result(test_case, agent);
                bool final_pass = success && validated;
                TestResult result;
                result.id = test_case.id;
                result.name = test_case.name;
                result.success = final_pass;
                result.llm_claimed_success = success;
                result.validation_passed = validated;
                result.duration = duration;
                result.details = details;
                result.output_snippet = output.substr(output.length() > 500 ? output.length() - 500 : 0);
                results.push_back(result);
                std::string status = final_pass ? "✓ PASS" : "✗ FAIL";
                std::cout << status << " in " << duration << "s\n";
                for(const auto& d : details) std::cout << "  " << d << "\n";
            } catch(const std::exception& e) {
                TestResult result;
                result.id = test_case.id;
                result.name = test_case.name;
                result.success = false;
                result.error = e.what();
                result.traceback = "Exception occurred";
                results.push_back(result);
                std::cout << "✗ FAIL - Exception: " << e.what() << "\n";
            }
        }
        int passed = std::count_if(results.begin(), results.end(), [](const auto& r) { return r.success; });
        int total = results.size();
        std::cout << "\n" << std::string(60, '=') << "\nSUMMARY: " << passed << "/" << total << " passed (" << (passed*100/total) << "%)\n";
        std::cout << "API Calls Avoided by Semantic Cache: " << agent->api_calls_avoided << "\n";
        std::cout << std::string(60, '=') << "\n";
        json report;
        json results_array = json::array();
        for (const auto& r : results) {
            json j;
            j["id"] = r.id;
            j["name"] = r.name;
            j["success"] = r.success;
            j["llm_claimed_success"] = r.llm_claimed_success;
            j["validation_passed"] = r.validation_passed;
            j["duration"] = r.duration;
            j["details"] = r.details;
            j["output_snippet"] = r.output_snippet;
            j["error"] = r.error;
            j["traceback"] = r.traceback;
            results_array.push_back(j);
        }
        report["results"] = results_array;
        report["summary"] = {{"passed", passed}, {"total", total}, {"cache_hits", agent->semantic_memory.get_cache_hits()}};
        std::string report_file = (suite_path / ("report_" + std::to_string((int)get_time()) + ".json")).string();
        std::ofstream f(report_file);
        f << report.dump(2);
        std::cout << "Report saved to: " << report_file << "\n";
        return report["summary"];
    }
};

int main() {
    curl_global_init(CURL_GLOBAL_ALL);
    srand(time(nullptr));
    try {
        AgentConfig agent_config;
        agent_config.max_execution_timeout = 45;
        agent_config.max_code_timeout = 20;
        agent_config.semantic_similarity_threshold = 0.55f;
        EvolutionConfig evolution_config;
        evolution_config.enabled = true;
        evolution_config.auto_reflect = true;
        evolution_config.max_context_turns = 6;
        CodingAgent agent(agent_config, evolution_config);
        TestSuite test_suite("./agent_test_suite");
        std::cout << "\nSelect mode:\n"
                  << "1. Run single task\n"
                  << "2. Run full test suite (T1-T10)\n"
                  << "3. Run specific test\n"
                  << "4. Cache stats\n"
                  << "5. Check environment\n\n"
                  << "Choice (1-5): ";
        std::string choice;
        std::getline(std::cin, choice);
        if(choice == "2") {
            auto results = test_suite.run_evaluation(&agent);
            std::cout << "\nFinal Score: " << results["passed"] << "/" << results["total"] << "\n";
        } else if(choice == "3") {
            std::cout << "Enter test ID (e.g., T1, T7, T10): ";
            std::string test_id;
            std::getline(std::cin, test_id);
            auto results = test_suite.run_evaluation(&agent, {test_id});
        } else if(choice == "4") {
            std::cout << "Total API calls avoided by semantic cache: " << agent.api_calls_avoided << "\n";
            std::cout << "Total cache hits (all time): " << agent.semantic_memory.get_cache_hits() << "\n";
        } else if(choice == "5") {
            std::cout << agent.get_environment_info() << "\n";
        } else {
            std::cout << "Enter task description (press Enter twice when finished):\n";
            std::string task, line;
            while(true) {
                std::getline(std::cin, line);
                if(line.empty()) break;
                task += line + " ";
            }
            if(task.empty()) {
                task = R"(Task 10. Concurrent File Hashing
                Write a script that scans a local directory and identifies all files with a .txt or .py extension. Use the
                concurrent.futures module to calculate the SHA-256 hash of each file in parallel using multiple threads or
                processes. The program should store the results in a dictionary where the keys are the filenames and the values
                are the computed hashes. Finally, print the total execution time and verify that the number of hashes matches
                the number of files found. This task tests concurrency, file system I/O, and the use of the hashlib library.)";
            }
            std::cout << "Max turns (default 6): ";
            std::string turns_str;
            std::getline(std::cin, turns_str);
            int max_turns = turns_str.empty() ? 6 : std::stoi(turns_str);
            std::string result = agent.run(task, max_turns);
            std::cout << "\n" << std::string(60, '=') << "\nFINAL RESULT\n" << std::string(60, '=') << "\n" << result << "\n";
        }
    } catch(const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        curl_global_cleanup();
        return 1;
    }
    curl_global_cleanup();
    return 0;
}
