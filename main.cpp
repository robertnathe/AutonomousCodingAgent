#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <regex>
#include <filesystem>
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <algorithm>
#include <array>
#include <ctime>
#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>
#include <curl/curl.h>
#include <set>
#include <iterator>
#include <limits>
#include <future>
#include <thread>
#include <mutex>
#include <queue>
#include <map>
#include <optional>
#include <variant>
#include <functional>

namespace fs = std::filesystem;
using json = nlohmann::json;

bool ends_with(const std::string& str, const std::string& suffix) {
    if (suffix.length() > str.length()) return false;
    return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
}

class HttpClient {
private:
    std::mutex curl_mutex_;
    static size_t write_callback(void* contents, size_t size, size_t nmemb, std::string* userp) {
        userp->append((char*)contents, size * nmemb);
        return size * nmemb;
    }
public:
    HttpClient() {
        curl_global_init(CURL_GLOBAL_DEFAULT);
    }
    ~HttpClient() {
        curl_global_cleanup();
    }
    std::string post(const std::string& url, const std::string& data, 
                     const std::vector<std::string>& headers, long timeout = 30L) {
        std::lock_guard<std::mutex> lock(curl_mutex_);
        CURL* curl = curl_easy_init();
        if (!curl) throw std::runtime_error("CURL not initialized");
        std::string response;
        struct curl_slist* header_list = nullptr;
        for (const auto& h : headers) {
            header_list = curl_slist_append(header_list, h.c_str());
        }
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, header_list);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
        CURLcode res = curl_easy_perform(curl);
        curl_slist_free_all(header_list);
        curl_easy_cleanup(curl);
        if (res != CURLE_OK) {
            throw std::runtime_error(std::string("HTTP request failed: ") + curl_easy_strerror(res));
        }
        return response;
    }
    std::string get(const std::string& url, const std::vector<std::string>& headers = {}, 
                    long timeout = 30L) {
        std::lock_guard<std::mutex> lock(curl_mutex_);
        CURL* curl = curl_easy_init();
        if (!curl) throw std::runtime_error("CURL not initialized");
        std::string response;
        struct curl_slist* header_list = nullptr;
        for (const auto& h : headers) {
            header_list = curl_slist_append(header_list, h.c_str());
        }
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, header_list);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout);
        CURLcode res = curl_easy_perform(curl);
        curl_slist_free_all(header_list);
        curl_easy_cleanup(curl);
        if (res != CURLE_OK) {
            throw std::runtime_error(std::string("HTTP GET failed: ") + curl_easy_strerror(res));
        }
        return response;
    }
};

struct QueryResult {
    std::vector<std::string> documents;
    std::vector<double> distances;
    std::vector<std::string> ids;
    std::vector<std::map<std::string, std::string>> metadatas;
};

class ChromaClient {
private:
    std::string base_url_ = "http://localhost:8000";
    std::string collection_name_ = "coding_agent_memory_v4";
    HttpClient http_;
    std::mutex memory_mutex_;

public:
    void ensure_collection() {
        std::lock_guard<std::mutex> lock(memory_mutex_);
        json create_req = {
            {"name", collection_name_}, 
            {"metadata", {{"description", "C++ Coding Agent Memory v4"}, {"version", "2.0"}}}
        };
        try { 
            http_.post(base_url_ + "/api/v1/collections", create_req.dump(), 
                      {"Content-Type: application/json"}); 
        } catch (...) {}
    }
    QueryResult query(const std::string& query_text, int n_results = 5, 
                     const std::map<std::string, std::string>& where_filter = {}) {
        std::lock_guard<std::mutex> lock(memory_mutex_);
        json query_req = {{"query_texts", {query_text}}, {"n_results", n_results}};
        if (!where_filter.empty()) {
            query_req["where"] = where_filter;
        }
        try {
            auto response = http_.post(base_url_ + "/api/v1/collections/" + collection_name_ + "/query", 
                                     query_req.dump(), {"Content-Type: application/json"});
            auto j = json::parse(response);
            QueryResult qr;
            if (j.contains("documents") && !j["documents"].empty()) {
                for (const auto& doc : j["documents"][0]) {
                    if (doc.is_string()) qr.documents.push_back(doc.get<std::string>());
                }
            }
            if (j.contains("distances") && !j["distances"].empty()) {
                for (const auto& d : j["distances"][0]) {
                    if (d.is_number()) qr.distances.push_back(d.get<double>());
                }
            }
            if (j.contains("ids") && !j["ids"].empty()) {
                for (const auto& id : j["ids"][0]) {
                    if (id.is_string()) qr.ids.push_back(id.get<std::string>());
                }
            }
            return qr;
        } catch (...) { return {}; }
    }
    void add_document(const std::string& id, const std::string& document, 
                     const std::map<std::string, std::string>& metadata = {}) {
        std::lock_guard<std::mutex> lock(memory_mutex_);
        json add_req = {{"ids", {id}}, {"documents", {document}}};
        if (!metadata.empty()) {
            add_req["metadatas"] = json::array({metadata});
        }
        try { 
            http_.post(base_url_ + "/api/v1/collections/" + collection_name_ + "/add", 
                      add_req.dump(), {"Content-Type: application/json"}); 
        } catch (...) {}
    }
    int count() {
        std::lock_guard<std::mutex> lock(memory_mutex_);
        try {
            auto response = http_.get(base_url_ + "/api/v1/collections/" + collection_name_ + "/count");
            auto j = json::parse(response);
            return j.value("count", 0);
        } catch (...) { return 0; }
    }
};

// ============================================================================
// GROQ AI CLIENT (OpenAI-compatible API)
// ============================================================================

//class GroqClient {
//private:
    //std::string api_key_;
    //std::string model_id_;
    //HttpClient http_;
//public:
    //GroqClient(const std::string& model_id = "llama-3.3-70b-versatile") : model_id_(model_id) {
        //const char* env_key = std::getenv("GROQ_API_KEY");
        //if (!env_key) throw std::runtime_error("Error: GROQ_API_KEY environment variable is not set.");
        //api_key_ = env_key;
    //}
    //std::string generate_content(const std::string& contents, const std::string& system_instruction, 
                                //float temperature = 0.0f, int max_tokens = 8192) {
        //std::string url = "https://api.groq.com/openai/v1/chat/completions";
        //// Build messages array with system and user messages
        //json messages = json::array();
        //if (!system_instruction.empty()) {
            //messages.push_back({{"role", "system"}, {"content", system_instruction}});
        //}
        //messages.push_back({{"role", "user"}, {"content", contents}});
        //json req = {
            //{"model", model_id_},
            //{"messages", messages},
            //{"temperature", temperature},
            //{"max_tokens", max_tokens}
        //};
        //std::vector<std::string> headers = {
            //"Content-Type: application/json",
            //"Authorization: Bearer " + api_key_
        //};
        //auto response = http_.post(url, req.dump(), headers, 60L);
        //auto j = json::parse(response);
        //// Groq uses OpenAI-compatible response format with "choices" array
        //if (j.contains("choices") && !j["choices"].empty() &&
            //j["choices"][0].contains("message") &&
            //j["choices"][0]["message"].contains("content")) {
            //return j["choices"][0]["message"]["content"].get<std::string>();
        //}
        //if (j.contains("error")) {
            //std::string error_msg = "API error";
            //if (j["error"].contains("message")) {
                //error_msg = j["error"]["message"].get<std::string>();
            //}
            //throw std::runtime_error("API error: " + error_msg);
        //}
        //throw std::runtime_error("Unexpected API response format");
    //}
//};

// ============================================================================
// GOOGLE AI CLIENT (Gemini API)
// ============================================================================

class GoogleClient {
private:
    std::string api_key_;
    std::string model_id_;
    HttpClient http_;
public:
    GoogleClient(const std::string& model_id = "gemini-2.5-flash") : model_id_(model_id) {
        const char* env_key = std::getenv("GOOGLE_API_KEY");
        if (!env_key) throw std::runtime_error("Error: GOOGLE_API_KEY environment variable is not set.");
        api_key_ = env_key;
    }
    std::string generate_content(const std::string& contents, const std::string& system_instruction, 
                                float temperature = 0.0f, int max_tokens = 8192) {
        std::string url = "https://generativelanguage.googleapis.com/v1beta/models/" + 
                          model_id_ + ":generateContent?key=" + api_key_;
        json request;
        request["contents"] = json::array({
            json{
                {"role", "user"},
                {"parts", json::array({json{{"text", contents}}})}
            }
        });
        if (!system_instruction.empty()) {
            request["system_instruction"] = json{
                {"parts", json::array({json{{"text", system_instruction}}})}
            };
        }
        request["generationConfig"] = json{
            {"temperature", temperature},
            {"maxOutputTokens", max_tokens}
        };
        std::vector<std::string> headers = {
            "Content-Type: application/json"
        };
        auto response = http_.post(url, request.dump(), headers, 60L);
        auto j = json::parse(response);
        if (j.contains("error")) {
            std::string error_msg = "API error";
            if (j["error"].contains("message")) {
                error_msg = j["error"]["message"].get<std::string>();
            }
            throw std::runtime_error("Google API error: " + error_msg);
        }
        if (j.contains("candidates") && !j["candidates"].empty() &&
            j["candidates"][0].contains("content") &&
            j["candidates"][0]["content"].contains("parts") &&
            !j["candidates"][0]["content"]["parts"].empty() &&
            j["candidates"][0]["content"]["parts"][0].contains("text")) {
            return j["candidates"][0]["content"]["parts"][0]["text"].get<std::string>();
        }
        throw std::runtime_error("Unexpected API response format: missing expected fields");
    }
};

enum class ActionType {
    READ_FILE,
    WRITE_FILE,
    LIST_FILES,
    EXECUTE_CPP,
    EXECUTE_PYTHON,
    EXECUTE_MULTI_CPP,
    UNDO_CHANGE,
    GET_ENV_INFO,
    VERIFY_RESULT,
    BATCH_OPERATIONS,
    CONDITIONAL,
    PARALLEL_GROUP
};

struct Action {
    std::string id;
    ActionType type;
    std::map<std::string, std::string> args;
    std::vector<std::string> depends_on;
    bool condition_required = false;
    std::string condition_expr;
    int retry_count = 0;
    int max_retries = 2;
    std::optional<std::string> result;
    bool executed = false;
    bool success = false;
    std::chrono::milliseconds execution_time{0};
};

struct ActionResult {
    std::string action_id;
    bool success;
    std::string output;
    std::string error;
    std::chrono::milliseconds duration;
};

class ActionExecutor {
private:
    std::map<std::string, ActionResult> results_;
    std::mutex results_mutex_;
    bool evaluate_condition(const std::string& expr, const std::map<std::string, ActionResult>& context) {
        if (expr.find(".success") != std::string::npos) {
            std::string action_id = expr.substr(0, expr.find(".success"));
            auto it = context.find(action_id);
            return it != context.end() && it->second.success;
        }
        if (expr.find(".output.contains") != std::string::npos) {
            size_t start = expr.find("(") + 1;
            size_t end = expr.find(")");
            std::string search_term = expr.substr(start, end - start);
            std::string action_id = expr.substr(0, expr.find(".output"));
            auto it = context.find(action_id);
            return it != context.end() && it->second.output.find(search_term) != std::string::npos;
        }
        return true;
    }
public:
    void store_result(const ActionResult& result) {
        std::lock_guard<std::mutex> lock(results_mutex_);
        results_[result.action_id] = result;
    }
    std::optional<ActionResult> get_result(const std::string& action_id) {
        std::lock_guard<std::mutex> lock(results_mutex_);
        auto it = results_.find(action_id);
        if (it != results_.end()) return it->second;
        return std::nullopt;
    }
    bool can_execute(const Action& action) {
        if (action.depends_on.empty()) return true;
        std::lock_guard<std::mutex> lock(results_mutex_);
        for (const auto& dep : action.depends_on) {
            auto it = results_.find(dep);
            if (it == results_.end() || !it->second.success) return false;
        }
        if (action.condition_required && !action.condition_expr.empty()) {
            return evaluate_condition(action.condition_expr, results_);
        }
        return true;
    }
    std::vector<std::string> get_ready_actions(const std::vector<Action>& pending) {
        std::vector<std::string> ready;
        for (const auto& action : pending) {
            if (!action.executed && can_execute(action)) {
                ready.push_back(action.id);
            }
        }
        return ready;
    }
};

class CodingAgent {
private:
    // GroqClient client_;
    GoogleClient client_;
    ChromaClient chroma_;
    std::string model_id_;
    std::string backup_dir_;
    std::string last_written_file_;
    ActionExecutor executor_;
    int action_counter_ = 0;
    struct SessionState {
        std::vector<std::string> created_files;
        std::vector<std::string> modified_files;
        std::map<std::string, std::string> file_checksums;
        int compilation_attempts = 0;
        int test_failures = 0;
        std::chrono::steady_clock::time_point start_time;
    } session_state_;
public:
    // CodingAgent(const std::string& model_id = "llama-3.3-70b-versatile")
    CodingAgent(const std::string& model_id = "gemini-2.5-flash")
        : client_(model_id), model_id_(model_id), backup_dir_(".agent_backups_v2") {
        fs::create_directories(backup_dir_);
        chroma_.ensure_collection();
        session_state_.start_time = std::chrono::steady_clock::now();
    }
    std::string backup_file(const std::string& file_path) {
        if (!fs::exists(file_path)) return "";
        fs::path backup_path = fs::path(backup_dir_) / 
                              (fs::path(file_path).filename().string() + "." + 
                               std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) + ".bak");
        fs::copy_file(file_path, backup_path, fs::copy_options::overwrite_existing);
        return backup_path.string();
    }
    std::vector<std::string> list_directory_files(const std::vector<std::string>& additional_ignore = {}) {
        std::vector<std::string> files;
        std::vector<std::string> ignore_patterns = {".git", "venv", "__pycache__", 
            ".agent_backups", ".agent_backups_v2", "build", "cmake-build-debug", ".idea", ".vscode"};
        ignore_patterns.insert(ignore_patterns.end(), additional_ignore.begin(), additional_ignore.end());
        for (const auto& entry : fs::recursive_directory_iterator(".", fs::directory_options::skip_permission_denied)) {
            if (!entry.is_regular_file()) continue;
            std::string path_str = entry.path().string();
            bool should_ignore = false;
            for (const auto& pattern : ignore_patterns) {
                if (path_str.find(pattern) != std::string::npos) { should_ignore = true; break; }
            }
            if (!should_ignore) files.push_back(fs::relative(entry.path(), ".").string());
        }
        return files;
    }
    std::string read_file(const std::string& file_path, int max_lines = 1000) {
        try {
            std::ifstream file(file_path);
            if (!file.is_open()) return "Error: Cannot open file " + file_path;
            std::stringstream buffer;
            std::string line;
            int line_count = 0;
            while (std::getline(file, line) && line_count < max_lines) {
                buffer << line << "\n";
                line_count++;
            }
            if (line_count >= max_lines) {
                buffer << "\n... [truncated, file exceeds " << max_lines << " lines] ...";
            }
            return buffer.str();
        } catch (const std::exception& e) {
            return std::string("Error reading file: ") + e.what();
        }
    }
    std::string write_file(const std::string& file_path, const std::string& content, 
                          bool create_backup = true) {
        if (file_path.empty() || file_path.find_first_not_of(" \t\n\r") == std::string::npos) {
            return "CRITICAL ERROR: invalid file_path!";
        }
        try {
            if (create_backup && fs::exists(file_path)) {
                backup_file(file_path);
                session_state_.modified_files.push_back(file_path);
            } else {
                session_state_.created_files.push_back(file_path);
            }
            fs::path p(file_path);
            if (p.has_parent_path()) fs::create_directories(p.parent_path());
            std::ofstream file(file_path);
            if (!file.is_open()) return "Error: Cannot open " + file_path + " for writing";
            file << content;
            file.flush();
            file.close();
            if (!fs::exists(file_path)) return "Error: File was not created";
            if (ends_with(file_path, ".cpp") || ends_with(file_path, ".cc") || 
                ends_with(file_path, ".cxx") || ends_with(file_path, ".py")) {
                last_written_file_ = file_path;
            }
            return "Successfully wrote " + std::to_string(content.size()) + " bytes to " + file_path;
        } catch (const std::exception& e) {
            return std::string("Error writing file: ") + e.what();
        }
    }
    std::string undo_last_change(const std::string& file_path) {
        fs::path dir(backup_dir_);
        if (!fs::exists(dir)) return "No backup directory found";
        std::vector<fs::path> backups;
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.is_regular_file() && entry.path().filename().string().find(file_path) != std::string::npos) {
                backups.push_back(entry.path());
            }
        }
        if (backups.empty()) return "No backup found for " + file_path;
        std::sort(backups.begin(), backups.end(), 
                 [](const fs::path& a, const fs::path& b) {
                     return fs::last_write_time(a) > fs::last_write_time(b);
                 });

        fs::copy_file(backups[0], file_path, fs::copy_options::overwrite_existing);
        return "Reverted " + file_path + " to backup from " + backups[0].filename().string();
    }
    std::string execute_multi_cpp(const std::vector<std::string>& sources, const std::string& output_name,
                                   const std::vector<std::string>& compile_flags = {"-std=c++17", "-Wall", "-O2"}) {
        session_state_.compilation_attempts++;
        std::string log_path = "./execution_traceback.log";
        try {
            std::ofstream(log_path, std::ios::trunc).close();
            if (sources.empty()) return "Error: No source files provided";
            std::string exec_name = output_name.empty() ? "program" : output_name;
#ifdef _WIN32
            exec_name += ".exe";
#endif
            std::string compile_cmd = "g++";
            for (const auto& flag : compile_flags) compile_cmd += " " + flag;
            compile_cmd += " -o \"" + exec_name + "\"";
            for (const auto& src : sources) compile_cmd += " \"" + src + "\"";
            compile_cmd += " 2>&1";
            auto compile_start = std::chrono::steady_clock::now();
            FILE* pipe = popen(compile_cmd.c_str(), "r");
            if (!pipe) return "Execution failed: Could not start compiler";
            std::string compile_out;
            char buf[4096];
            while (fgets(buf, sizeof(buf), pipe)) compile_out += buf;
            int compile_rc = pclose(pipe);
            auto compile_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - compile_start);
            if (compile_rc != 0) {
                std::ofstream f(log_path, std::ios::app);
                f << "=== COMPILATION FAILED ===\n";
                f << "Command: " << compile_cmd << "\n";
                f << "Duration: " << compile_duration.count() << "ms\n";
                f << "Output:\n" << compile_out << "\n";
                return "COMPILATION FAILED (rc=" + std::to_string(compile_rc) + 
                       ", time=" + std::to_string(compile_duration.count()) + "ms):\n" + compile_out;
            }
            auto run_start = std::chrono::steady_clock::now();
            std::string run_cmd = "./\"" + exec_name + "\" 2>&1";
            pipe = popen(run_cmd.c_str(), "r");
            if (!pipe) return "Execution failed: Could not run program";
            std::string run_out;
            while (fgets(buf, sizeof(buf), pipe)) run_out += buf;
            int run_rc = pclose(pipe);
            auto run_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - run_start);
            std::ofstream f(log_path, std::ios::app);
            std::time_t now = std::time(nullptr);
            f << "=== Execution (" << exec_name << ") ===\n";
            f << "Time: " << std::ctime(&now);
            f << "Compile time: " << compile_duration.count() << "ms\n";
            f << "Run time: " << run_duration.count() << "ms\n";
            f << "Return code: " << run_rc << "\n";
            f << "Output:\n" << run_out << "\n";
            std::string result = "✓ COMPILATION SUCCESS (" + std::to_string(compile_duration.count()) + "ms)\n";
            result += "✓ EXECUTION (rc=" + std::to_string(run_rc) + ", " + std::to_string(run_duration.count()) + "ms)\n";
            result += "Output:\n" + (run_out.empty() ? "(no output)" : run_out);
            return result;
        } catch (const std::exception& e) {
            return "Execution failed: " + std::string(e.what());
        }
    }
    std::string execute_cpp_file(const std::string& file_path, const std::vector<std::string>& extra_sources = {}) {
        std::vector<std::string> sources;
        std::istringstream iss(file_path);
        std::string token;
        while (iss >> token) sources.push_back(token);
        sources.insert(sources.end(), extra_sources.begin(), extra_sources.end());
        if (sources.empty()) return "Error: No source files";
        std::string exec_name = sources[0];
        size_t dot = exec_name.rfind('.');
        if (dot != std::string::npos) exec_name = exec_name.substr(0, dot);
        return execute_multi_cpp(sources, exec_name);
    }
    std::string execute_python_file(const std::string& file_path, const std::vector<std::string>& args = {}) {
        std::string log_path = "./execution_traceback.log";
        try {
            std::ofstream(log_path, std::ios::trunc).close();
            std::ostringstream cmd;
            cmd << "python3 \"" << file_path << "\"";
            for (const auto& arg : args) {
                cmd << " \"" << arg << "\"";
            }
            cmd << " 2>&1";
            std::string cmd_str = cmd.str();
            auto start = std::chrono::steady_clock::now();
            FILE* pipe = popen(cmd_str.c_str(), "r");
            if (!pipe) return "Execution failed: Could not start Python";
            std::string output;
            char buffer[4096];
            while (fgets(buffer, sizeof(buffer), pipe)) output += buffer;
            int status = pclose(pipe);
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start);
            std::ofstream f(log_path, std::ios::app);
            std::time_t now = std::time(nullptr);
            f << "=== Python Execution (" << file_path << ") ===\n";
            f << "Time: " << std::ctime(&now);
            f << "Duration: " << duration.count() << "ms\n";
            f << "Return code: " << status << "\n";
            f << "Output:\n" << output << "\n";
            if (status != 0) session_state_.test_failures++;
            return "Return code: " + std::to_string(status) + 
                   " (" + std::to_string(duration.count()) + "ms)\n" +
                   "Output:\n" + (output.empty() ? "(none)" : output);
        } catch (const std::exception& e) {
            return "Execution failed: " + std::string(e.what());
        }
    }
    std::string get_environment_info() {
        auto run_cmd = [](const std::string& cmd) -> std::string {
            FILE* pipe = popen(cmd.c_str(), "r");
            if (!pipe) return "(failed)";
            char buffer[256];
            std::string out;
            while (fgets(buffer, sizeof(buffer), pipe)) out += buffer;
            pclose(pipe);
            return out.empty() ? "(empty)" : out;
        };
        std::string info = "=== Environment Info ===\n";
        info += "C++ Compiler: " + run_cmd("g++ --version | head -1") + "\n";
        info += "Python: " + run_cmd("python3 --version") + "\n";
        info += "Working Directory: " + fs::current_path().string() + "\n";
        info += "\nPython Packages:\n" + run_cmd("python3 -m pip list 2>/dev/null | head -20") + "...";
        return info;
    }
    std::string retrieve_memories(const std::string& task_description, int n_results = 5) {
        auto query_res = chroma_.query(task_description, n_results);
        if (query_res.documents.empty()) return "No prior relevant memories.";
        std::string memory = "=== Relevant Past Experiences ===\n";
        for (size_t i = 0; i < query_res.documents.size(); ++i) {
            memory += "[Similarity: " + std::to_string(1.0 - query_res.distances[i]) + "]\n";
            memory += query_res.documents[i].substr(0, 500) + "\n---\n";
        }
        return memory;
    }
    void store_episode(const std::string& task, const std::string& final_log, 
                      const std::string& outcome = "unknown") {
        int count = chroma_.count();
        std::string doc = "Task: " + task + "\n";
        doc += "Outcome: " + outcome + "\n";
        doc += "Result: " + final_log.substr(0, 2000);
        auto now = std::chrono::system_clock::now();
        std::string id = "ep_" + std::to_string(count) + "_" + 
                        std::to_string(now.time_since_epoch().count());
        std::map<std::string, std::string> metadata;
        metadata["outcome"] = outcome;
        metadata["timestamp"] = std::to_string(std::time(nullptr));
        metadata["model"] = model_id_;
        chroma_.add_document(id, doc, metadata);
    }
    std::vector<Action> parse_actions(const std::string& text) {
        std::vector<Action> actions;
        std::regex actions_pattern(R"(```actions\s*([\s\S]*?)\s*```)", std::regex::icase);
        std::smatch match;
        if (!std::regex_search(text, match, actions_pattern)) {
            return actions;
        }
        std::string yaml_text = match[1].str();
        try {
            YAML::Node root = YAML::Load(yaml_text);
            if (root.IsSequence()) {
                for (const auto& node : root) {
                    actions.push_back(parse_single_action(node));
                }
            } else if (root.IsMap()) {
                actions.push_back(parse_single_action(root));
            }
        } catch (const YAML::Exception& e) {
            std::cerr << "YAML parsing error: " << e.what() << std::endl;
        }
        return actions;
    }
    Action parse_single_action(const YAML::Node& node) {
        Action action;
        action.id = "action_" + std::to_string(++action_counter_);
        if (node["id"]) action.id = node["id"].as<std::string>();
        if (node["tool"]) {
            std::string tool = node["tool"].as<std::string>();
            action.type = string_to_action_type(tool);
        }
        if (node["args"] && node["args"].IsMap()) {
            for (const auto& kv : node["args"]) {
                std::string key = kv.first.as<std::string>();
                std::string value = kv.second.as<std::string>();
                action.args[key] = value;
            }
        }
        if (node["depends_on"] && node["depends_on"].IsSequence()) {
            for (const auto& dep : node["depends_on"]) {
                action.depends_on.push_back(dep.as<std::string>());
            }
        }
        if (node["condition"]) {
            action.condition_required = true;
            action.condition_expr = node["condition"].as<std::string>();
        }
        if (node["retry"]) action.max_retries = node["retry"].as<int>();

        return action;
    }
    ActionType string_to_action_type(const std::string& tool) {
        if (tool == "read_file") return ActionType::READ_FILE;
        if (tool == "write_file") return ActionType::WRITE_FILE;
        if (tool == "list_directory_files") return ActionType::LIST_FILES;
        if (tool == "execute_cpp_file") return ActionType::EXECUTE_CPP;
        if (tool == "execute_python_file") return ActionType::EXECUTE_PYTHON;
        if (tool == "execute_multi_cpp") return ActionType::EXECUTE_MULTI_CPP;
        if (tool == "undo_last_change") return ActionType::UNDO_CHANGE;
        if (tool == "get_environment_info") return ActionType::GET_ENV_INFO;
        if (tool == "verify_result") return ActionType::VERIFY_RESULT;
        if (tool == "batch_operations") return ActionType::BATCH_OPERATIONS;
        return ActionType::READ_FILE;
    }
    std::string execute_action(const Action& action) {
        auto start = std::chrono::steady_clock::now();
        std::string result;
        switch (action.type) {
            case ActionType::READ_FILE:
                if (action.args.count("file_path")) {
                    int max_lines = action.args.count("max_lines") ? 
                                   std::stoi(action.args.at("max_lines")) : 1000;
                    result = read_file(action.args.at("file_path"), max_lines);
                } else result = "Error: missing file_path";
                break;
            case ActionType::WRITE_FILE:
                if (action.args.count("file_path") && action.args.count("content")) {
                    bool backup = !action.args.count("no_backup");
                    result = write_file(action.args.at("file_path"), 
                                      action.args.at("content"), backup);
                } else result = "Error: missing file_path or content";
                break;
            case ActionType::LIST_FILES: {
                result = "Files: ";
                auto files = list_directory_files();
                for (size_t i = 0; i < files.size(); ++i) {
                    if (i > 0) result += ", ";
                    result += files[i];
                }
                break;
            }
            case ActionType::EXECUTE_CPP:
                if (action.args.count("file_path")) {
                    std::vector<std::string> extras;
                    if (action.args.count("extra_sources")) {
                        std::istringstream iss(action.args.at("extra_sources"));
                        std::string src;
                        while (iss >> src) extras.push_back(src);
                    }
                    result = execute_cpp_file(action.args.at("file_path"), extras);
                } else result = "Error: missing file_path";
                break;
            case ActionType::EXECUTE_PYTHON:
                if (action.args.count("file_path")) {
                    std::vector<std::string> args;
                    if (action.args.count("args")) {
                        std::istringstream iss(action.args.at("args"));
                        std::string arg;
                        while (iss >> arg) args.push_back(arg);
                    }
                    result = execute_python_file(action.args.at("file_path"), args);
                } else result = "Error: missing file_path";
                break;
            case ActionType::EXECUTE_MULTI_CPP:
                if (action.args.count("sources")) {
                    std::vector<std::string> sources;
                    std::istringstream iss(action.args.at("sources"));
                    std::string src;
                    while (iss >> src) sources.push_back(src);
                    std::string output = action.args.count("output") ? 
                                        action.args.at("output") : "";
                    result = execute_multi_cpp(sources, output);
                } else result = "Error: missing sources";
                break;
            case ActionType::UNDO_CHANGE:
                if (action.args.count("file_path")) {
                    result = undo_last_change(action.args.at("file_path"));
                } else result = "Error: missing file_path";
                break;
            case ActionType::GET_ENV_INFO:
                result = get_environment_info();
                break;
            default:
                result = "Unknown action type";
        }
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start);
        ActionResult ar;
        ar.action_id = action.id;
        ar.output = result;
        ar.success = result.find("Error") == std::string::npos && 
                     result.find("FAILED") == std::string::npos;
        ar.duration = duration;
        executor_.store_result(ar);
        return "[" + action.id + " | " + std::to_string(duration.count()) + "ms] " + result;
    }
    std::vector<std::string> execute_actions_parallel(std::vector<Action>& actions) {
        std::vector<std::string> results;
        std::vector<std::future<std::pair<std::string, ActionResult>>> futures;
        std::map<int, std::vector<Action*>> levels;
        int current_level = 0;
        std::set<std::string> completed;
        while (completed.size() < actions.size()) {
            bool progress = false;
            for (auto& action : actions) {
                if (action.executed) continue;
                bool ready = true;
                for (const auto& dep : action.depends_on) {
                    if (completed.find(dep) == completed.end()) {
                        ready = false; break;
                    }
                }
                if (ready && executor_.can_execute(action)) {
                    levels[current_level].push_back(&action);
                    action.executed = true;
                    progress = true;
                }
            }
            if (!progress && completed.size() < actions.size()) {
                break;
            }
            current_level++;
        }
        for (auto& [level, level_actions] : levels) {
            futures.clear();
            for (auto* action : level_actions) {
                futures.push_back(std::async(std::launch::async, [this, action]() {
                    std::string result = execute_action(*action);
                    auto opt_ar = executor_.get_result(action->id);
                    return std::make_pair(result, opt_ar.value_or(ActionResult{}));
                }));
            }
            for (auto& fut : futures) {
                auto [result, ar] = fut.get();
                results.push_back(result);
                if (ar.success) completed.insert(ar.action_id);
            }
        }
        return results;
    }
    std::vector<std::string> execute_actions_sequential(std::vector<Action>& actions) {
        std::vector<std::string> results;
        std::set<std::string> completed;
        int max_iterations = actions.size() * 2;
        int iteration = 0;
        while (completed.size() < actions.size() && iteration < max_iterations) {
            bool progress = false;
            for (auto& action : actions) {
                if (action.executed) continue;
                bool deps_satisfied = true;
                for (const auto& dep : action.depends_on) {
                    if (completed.find(dep) == completed.end()) {
                        deps_satisfied = false; break;
                    }
                }
                if (deps_satisfied && executor_.can_execute(action)) {
                    std::string result = execute_action(action);
                    results.push_back(result);
                    action.executed = true;
                    auto opt_ar = executor_.get_result(action.id);
                    if (opt_ar && opt_ar->success) {
                        completed.insert(action.id);
                    }
                    progress = true;
                    if (opt_ar && !opt_ar->success && action.retry_count < action.max_retries) {
                        action.retry_count++;
                        action.executed = false;
                        std::cout << "  [Retry " << action.retry_count << "/" << action.max_retries 
                                 << " for " << action.id << "]" << std::endl;
                    }
                }
            }
            if (!progress) {
                std::cerr << "Warning: Action execution stalled, possible dependency cycle" << std::endl;
                break;
            }
            iteration++;
        }
        return results;
    }
    std::string run(const std::string& task_description, int max_turns = 5) {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "STARTING TASK: " << task_description.substr(0, 100) << "..." << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        std::string memory = retrieve_memories(task_description, 5);
        auto query_res = chroma_.query(task_description, 3, {{"outcome", "success"}});
        if (!query_res.documents.empty() && query_res.distances[0] < 0.3) {
            std::cout << "\n[Memory Reuse] Found similar successful solution (similarity: " 
                     << (1.0 - query_res.distances[0]) << ")" << std::endl;
            std::string adaptation_prompt = "Adapt this successful solution to the new task:\n" +
                                          query_res.documents[0] + "\n\nNew Task: " + task_description;
            try {
                std::string adapted = client_.generate_content(adaptation_prompt, 
                    "Output only the adapted actions YAML block.", 0.1f, 4096);
                auto adapted_actions = parse_actions(adapted);
                if (!adapted_actions.empty()) {
                    std::cout << "[Memory Reuse] Executing adapted solution with " 
                             << adapted_actions.size() << " actions" << std::endl;
                    auto results = execute_actions_sequential(adapted_actions);
                    if (verify_task_completion(task_description, results)) {
                        store_episode(task_description, "Memory-adapted solution", "success");
                        return "Task completed via memory adaptation.\n" + join_results(results);
                    }
                }
            } catch (...) {
                std::cout << "[Memory Reuse] Adaptation failed, proceeding with fresh solution" << std::endl;
            }
        }
        const std::string system_prompt = build_system_prompt();
        std::vector<std::string> session_context;
        std::string final_result;
        for (int turn = 0; turn < max_turns; ++turn) {
            auto turn_start = std::chrono::steady_clock::now();
            std::cout << "\n--- TURN " << (turn + 1) << "/" << max_turns << " ---" << std::endl;
            std::string full_prompt = build_full_prompt(task_description, memory, session_context, turn);
            try {
                std::string response = client_.generate_content(full_prompt, system_prompt, 0.2f, 8192);
                auto actions = parse_actions(response);
                if (actions.empty()) {
                    std::cout << "[Warning] No actions found in response" << std::endl;
                    session_context.push_back("Turn " + std::to_string(turn + 1) + 
                                               ": No actions generated");
                    continue;
                }
                std::cout << "[Actions] Found " << actions.size() << " action(s) to execute" << std::endl;
                bool has_dependencies = false;
                for (const auto& a : actions) {
                    if (!a.depends_on.empty() || a.condition_required) has_dependencies = true;
                }
                std::vector<std::string> observations;
                if (has_dependencies || actions.size() > 3) {
                    std::cout << "[Strategy] Using sequential execution with dependency resolution" << std::endl;
                    observations = execute_actions_sequential(actions);
                } else {
                    std::cout << "[Strategy] Using parallel execution" << std::endl;
                    observations = execute_actions_parallel(actions);
                }
				bool success_declared = check_success_declaration(response);
				std::string obs_summary = "OBSERVATIONS:\n" + join_results(observations);
				std::cout << obs_summary << std::endl;
				bool verified = verify_task_completion(task_description, observations);
				if (success_declared || verified) {
				    std::cout << "\n[Verification] " 
				              << (verified ? "Auto-verified success" : "Success declared") 
				              << " - cleaning up test files..." << std::endl;
				    cleanup_test_files();
				    std::cout << "✓ VERIFICATION PASSED" << std::endl;
				    store_episode(task_description, response + "\n" + obs_summary, "success");
				    return response + "\n" + obs_summary + "\n[Automatic cleanup of test files performed]";
				} else if (success_declared) {
				    std::cout << "✗ VERIFICATION FAILED - continuing..." << std::endl;
				    session_context.push_back("Turn " + std::to_string(turn + 1) + 
				                              " claimed success but verification failed");
				}                
                auto turn_duration = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now() - turn_start);
                session_context.push_back("Turn " + std::to_string(turn + 1) + " (" + 
                                        std::to_string(turn_duration.count()) + "s):\n" +
                                        response + "\n" + obs_summary);
                if (turn > 5) {
                    bool making_progress = check_progress(session_context);
                    if (!making_progress) {
                        std::cout << "[Warning] No progress detected, injecting hint..." << std::endl;
                        session_context.push_back("SYSTEM: Multiple turns without progress. "
                                                "Consider a different approach or check file contents.");
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "Error in turn " << (turn + 1) << ": " << e.what() << std::endl;
                session_context.push_back("Turn " + std::to_string(turn + 1) + " ERROR: " + e.what());
            }
        }
        store_episode(task_description, join_results(session_context), "incomplete");
        return "Task incomplete after " + std::to_string(max_turns) + " turns.\n" + 
               join_results(session_context);
    }
    
private:
    void cleanup_test_files() {
        std::cout << "[Auto-Cleanup] Removing temporary test artifacts..." << std::endl;
        std::string main_script = last_written_file_;
        if (main_script.empty() || !ends_with(main_script, ".py")) {
            auto files = list_directory_files();
            for (const auto& f : files) {
                if (ends_with(f, ".py")) {
                    main_script = f;
                    break;
                }
            }
        }
        for (const auto& entry : fs::recursive_directory_iterator(".", 
             fs::directory_options::skip_permission_denied)) {
            if (!entry.is_regular_file()) continue;
            std::string p = fs::relative(entry.path(), ".").string();
            if (p == main_script) continue;
            if (ends_with(p, ".txt") ||
                (ends_with(p, ".py") && p.find("test") != std::string::npos)) {
                try {
                    fs::remove(entry.path());
                    std::cout << "  Deleted: " << p << std::endl;
                } catch (...) {}
            }
        }
        std::cout << "[Auto-Cleanup] Complete." << std::endl;
    }    
    
private:
    std::string build_system_prompt() {
        return R"(You are an AI Coding Agent with filesystem and execution capabilities.

AVAILABLE TOOLS:
- read_file(file_path: str, max_lines: int = 1000)
- write_file(file_path: str, content: str, no_backup: bool = false)
- list_directory_files()
- execute_cpp_file(file_path: str, extra_sources: str = "")
- execute_python_file(file_path: str, args: str = "")
- execute_multi_cpp(sources: str, output: str = "")
- undo_last_change(file_path: str)
- get_environment_info()

ACTION FORMAT:
Always respond with YAML actions block:
```actions
- id: action_1
  tool: read_file
  args:
    file_path: main.cpp
    max_lines: 50
- id: action_2
  tool: write_file
  args:
    file_path: main.cpp
    content: |
      #include <iostream>
      int main() { std::cout << "Hello"; }
  depends_on: [action_1]
- id: action_3
  tool: execute_cpp_file
  args:
    file_path: main.cpp
  depends_on: [action_2]
  condition: action_2.success
```

MULTI-ACTION STRATEGY:
1. Plan: Analyze the task and break it into logical steps
2. Batch: Group independent operations (e.g., read multiple files simultaneously)
3. Chain: Use depends_on to sequence dependent operations
4. Verify: Always include execution/verification as final step
5. Adapt: If execution fails, analyze error and retry with fixes

CRITICAL RULES:
- ALWAYS verify code compiles and runs before declaring success
- Use depends_on to ensure proper execution order
- For multi-file projects, list all sources in execute_multi_cpp
- Read existing files before modifying them
- Include error handling and edge cases in generated code
- After write_file, always execute to verify

OUTCOME FORMAT:
When task is complete and verified:
```yaml
outcome: success
verification: <how you verified the solution works>
```

When task needs more work:
```yaml
outcome: continue
reason: <why more work is needed>
next_steps: <what should be done next>
```
)";
    }
    std::string build_full_prompt(const std::string& task, const std::string& memory,
                                  const std::vector<std::string>& context, int turn) {
        std::string prompt = "LONG-TERM MEMORY:\n" + memory + "\n\n";
        if (!context.empty()) {
            prompt += "SESSION HISTORY (last 3 turns):\n";
            int start = std::max(0, (int)context.size() - 3);
            for (size_t i = start; i < context.size(); ++i) {
                prompt += context[i] + "\n---\n";
            }
            prompt += "\n";
        }
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - session_state_.start_time);
        prompt += "SESSION STATS:\n";
        prompt += "- Turn: " + std::to_string(turn + 1) + "\n";
        prompt += "- Elapsed: " + std::to_string(elapsed.count()) + "s\n";
        prompt += "- Files created: " + std::to_string(session_state_.created_files.size()) + "\n";
        prompt += "- Files modified: " + std::to_string(session_state_.modified_files.size()) + "\n";
        prompt += "- Compilation attempts: " + std::to_string(session_state_.compilation_attempts) + "\n";
        prompt += "- Test failures: " + std::to_string(session_state_.test_failures) + "\n\n";
        prompt += "CURRENT TASK:\n" + task + "\n\n";
        if (turn == 0) {
            prompt += "INSTRUCTION: Analyze the task and create a step-by-step plan. ";
            prompt += "For the first turn, focus on understanding existing files and creating the initial implementation.";
        } else {
            prompt += "INSTRUCTION: Review previous results. If there were errors, fix them. ";
            prompt += "If successful, verify thoroughly. Continue until task is complete.";
        }
        return prompt;
    }
	bool check_success_declaration(const std::string& response) {
	    return response.find("outcome: success") != std::string::npos;
	}
    bool verify_task_completion(const std::string& task, const std::vector<std::string>& observations) {
        std::regex file_regex(R"(\b[\w-]+\.(?:hpp|cpp|cc|cxx|py)\b)", std::regex::icase);
        std::vector<std::string> required_files;
        auto begin = std::sregex_iterator(task.begin(), task.end(), file_regex);
        for (auto i = begin; i != std::sregex_iterator(); ++i) {
            required_files.push_back(i->str());
        }
        for (const auto& f : required_files) {
            if (!fs::exists(f)) {
                std::cout << "  [Verify] Missing required file: " << f << std::endl;
                return false;
            }
        }
        bool has_success = false;
        for (const auto& obs : observations) {
            if (obs.find("✓ COMPILATION SUCCESS") != std::string::npos ||
                obs.find("Return code: 0") != std::string::npos) {
                has_success = true;
            }
            if (obs.find("FAILED") != std::string::npos || 
                obs.find("Error:") != std::string::npos) {
                std::cout << "  [Verify] Found error in observations" << std::endl;
                return false;
            }
        }
        return has_success || required_files.empty();
    }
    bool check_progress(const std::vector<std::string>& context) {
        if (context.size() < 3) return true;
        bool progress = false;
        for (size_t i = context.size() - 3; i < context.size(); ++i) {
            if (i < 0) continue;
            const auto& ctx = context[i];
            if (ctx.find("Successfully wrote") != std::string::npos ||
                ctx.find("COMPILATION SUCCESS") != std::string::npos ||
                ctx.find("Return code: 0") != std::string::npos) {
                progress = true;
            }
        }
        return progress;
    }
    std::string join_results(const std::vector<std::string>& results) {
        std::string joined;
        for (size_t i = 0; i < results.size(); ++i) {
            if (i > 0) joined += "\n";
            joined += results[i];
        }
        return joined;
    }
};

int main() {
    try {
        CodingAgent agent;

        //std::string task = R"(
        // Task 1
//Create a script prime_factor.py that finds the largest prime factor of the number 600851475143. Start with 
//a basic trial division approach, but if the execution exceeds one second, refactor the code to use an optimized 
//square root limit $O(\sqrt{n})$ algorithm. Print the final result and the time taken for the calculation.
//)";
        //std::string result = agent.run(task, 10);
        // Run test correct
        
        //std::string task = R"(
        // Task 2
//Implement a two-step structural change to test directory and import awareness:
    //1. Create a directory math_lib/ with a file operations.py containing a function power(a, b). Then, create 
       //app.py in the root directory that imports this function and prints the result of $2^{10}$.
    //2. Move operations.py into a new nested directory structure core/utils/, update the import statements in app.py 
       //to reflect the change, and run app.py to verify the output remains 1024.
//)";
        //std::string result = agent.run(task, 10);
        // Run test correct

        //std::string task = R"(
        // Task 3
//Create a vending machine simulation in vending.py. The script must define a VendingMachine class with a product 
//catalog and a custom exception InsufficientFundsError. Demonstrate the following logic:
    //1. Deposit 2.00 and attempt to buy an item priced at 2.50.
    //2. Catch the InsufficientFundsError and print a custom message.
    //3. Deposit an additional 1.00, successfully purchase the item, and print the remaining balance and updated inventory count.
//)";
        //std::string result = agent.run(task, 10);
        // Run test correct

        //std::string task = R"(
        // Task 4
//Create a data processing pipeline that handles dirty data:
    //1. Write a CSV file raw_data.csv with three columns: Name, Age, and Salary. Include at least 6 rows, where some 
       //Age values are strings (e.g., twenty) and some Salary values are empty.
    //2. Create cleaner.py to read this CSV, remove rows with missing or non-numeric data, and calculate the average 
       //Salary of the remaining valid entries.
    //3. Save the final average and the list of valid names to a JSON file named processed_data.json.
//)";
        //std::string result = agent.run(task, 10);
        // Run test correct

        //std::string task = R"(
        // Task 5
//Write a script pi_approx.py to approximate the value of $\pi$ using the Leibniz formula:
//$$\pi = 4 \sum_{n=0}^{\infty} \frac{(-1)^n}{2n+1}$$
//The agent must implement this as an iterative loop that continues until the absolute difference between the approximation 
//and math.pi is less than $10^{-5}$. Print the total number of iterations required to reach this level of precision.
//)";
        //std::string result = agent.run(task, 10);
        // Run test correct
        
        //std::string task = R"(
		//Task 6. Shortest Path in a Grid
		//Create a program that represents a 10×10 grid as a graph where specific cells are designated as obstacles. Implement a 
		//Breadth-First Search algorithm to find the shortest path from the coordinate (0,0) to (9,9). The script should output the 
		//sequence of coordinates representing the path and the total number of steps. If no path is possible, the program should 
		//catch the condition and print a descriptive message. This task tests the ability to handle graph traversal, state management, 
		//and grid-based logic.
//)";
        //std::string result = agent.run(task, 10);
        // Run test correct
        
        //std::string task = R"(
		//Task 7. Relational Database Management with SQLite
		//Develop a program that initializes an SQLite database with two related tables: Departments and Employees. The Departments table 
		//should contain a DepartmentID and DepartmentName. The Employees table should contain an EmployeeID, Name, Salary, and a 
		//DepartmentID as a foreign key. Insert at least three departments and ten employees into the database. Write a query using a JOIN 
		//statement to retrieve all employees who earn more than the average salary of their respective department. Save this filtered list 
		//into a new file named high_earners.json. This task evaluates database interaction, SQL syntax, and data relational mapping.
//)";
        //std::string result = agent.run(task, 10);
        // Run test correct

        //std::string task = R"(
		//Task 8. Memoization and Custom Decorators
		//Implement a custom decorator named memoize that caches the results of function calls to improve performance for expensive 
		//computations. Apply this decorator to a recursive function that calculates the n-th term of the Fibonacci sequence. The program 
		//must demonstrate the efficiency gain by calculating F50​ and printing the result along with the time taken. For comparison, the 
		//script should also attempt to calculate a smaller term like F30​ without the decorator and display the difference in execution 
		//speed. This task evaluates higher-order functions, closures, and algorithmic optimization.
//)";
        //std::string result = agent.run(task, 10);
        // Run test correct

        //std::string task = R"(
		//Task 9. Monte Carlo Integration
		//Create a program to estimate the area under the curve f(x)=sin(x) between x=0 and x=π using a Monte Carlo simulation. The 
		//program should generate 1,000,000 random points within a bounding box [0,π]×[0,1] and determine the fraction of points that 
		//fall below the curve. Compare the estimated result to the analytical value of 2 and print the absolute error. This task tests 
		//the use of random number generation, mathematical functions, and the ability to implement a probabilistic numerical method.
//)";
        //std::string result = agent.run(task, 10);
        // Run test correct
        
        std::string task = R"(
		Task 10. Concurrent File Hashing
		Write a script that scans a local directory and identifies all files with a .txt or .py extension. Use the concurrent.futures 
		module to calculate the SHA-256 hash of each file in parallel using multiple threads or processes. The program should store the 
		results in a dictionary where the keys are the filenames and the values are the computed hashes. Finally, print the total 
		execution time and verify that the number of hashes matches the number of files found. This task tests concurrency, file system 
		I/O, and the use of the hashlib library.
)";
        std::string result = agent.run(task, 5);
        // Run test correct

        std::cout << "FINAL RESULT:\n" << result << std::endl;
        std::cout << std::string(60, '=') << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
