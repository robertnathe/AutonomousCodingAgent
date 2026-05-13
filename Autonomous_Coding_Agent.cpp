// You are an advanced AI coding agent specializing in algorithmic discovery and optimization, powered by cutting-edge LLMs.
// Your mission is to take an initial C++ algorithm, analyze its purpose, understand simulated evaluation results, and propose
// a single significant improvement.

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <mutex>
#include <optional>
#include <random>
#include <regex>
#include <set>
#include <signal.h>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <tuple>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <openssl/md5.h>

using namespace std;
namespace fs = std::filesystem;
using json = nlohmann::json;

string trim(const string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

vector<string> split(const string& s, char d) {
    vector<string> r;
    stringstream ss(s);
    string p;
    while (getline(ss, p, d)) {
        p = trim(p);
        if (!p.empty()) r.push_back(p);
    }
    return r;
}

string md5_hash(const string& s) {
    unsigned char digest[MD5_DIGEST_LENGTH];
    MD5((const unsigned char*)s.c_str(), s.size(), digest);
    ostringstream oss;
    for (int i = 0; i < 4; ++i)
        oss << hex << setw(2) << setfill('0') << (int)digest[i];
    return oss.str();
}

double get_time() {
    return chrono::duration<double>(chrono::system_clock::now().time_since_epoch()).count();
}

const unordered_set<string> STOP_WORDS = {
    "the","and","is","in","to","of","a","for","on","with","as","by","at","an","be","this","that",
    "it","not","or","but","are","from","has","had","have","will","would","could","should","may","can",
    "do","does","did","was","were","been","being","am","i","you","he","she","we","they","me","him",
    "her","us","them","my","your","his","its","our","their","what","when","where","why","how","all",
    "any","both","each","few","more","most","other","some","such","no","nor","only","own","same","so",
    "than","through","too","under","until","up","very","with","would"
};

vector<string> tokenize(const string& text) {
    string lo = text;
    transform(lo.begin(), lo.end(), lo.begin(), ::tolower);
    regex wr(R"(\b[a-z]{3,}\b)");
    vector<string> w;
    for (sregex_iterator it(lo.begin(), lo.end(), wr), end; it != end; ++it) {
        string x = it->str();
        if (!STOP_WORDS.count(x)) w.push_back(x);
    }
    return w;
}

vector<pair<string,double>> compute_tfidf_vector(const string& text,
    const unordered_map<string,double>& idf, int total_docs) {
    auto tokens = tokenize(text);
    unordered_map<string,int> freq;
    for (const auto& t : tokens) freq[t]++;
    if (freq.empty()) return {};
    int mf = 0;
    for (const auto& [k,v] : freq) mf = max(mf, v);
    vector<pair<string,double>> vec;
    for (const auto& [term, cnt] : freq) {
        double tf = (double)cnt / mf;
        double idf_val = 1.0;
        auto it = idf.find(term);
        if (it != idf.end()) idf_val = it->second;
        else if (total_docs > 0) idf_val = log(1.0 + total_docs + 1);
        vec.emplace_back(term, tf * idf_val);
    }
    sort(vec.begin(), vec.end(), [](auto& a, auto& b){ return a.first < b.first; });
    return vec;
}

double cosine_similarity(const vector<pair<string,double>>& v1,
                         const vector<pair<string,double>>& v2, double v2_norm) {
    double dot = 0.0, n1 = 0.0;
    size_t i = 0, j = 0;
    while (i < v1.size() && j < v2.size()) {
        if (v1[i].first == v2[j].first) {
            dot += v1[i].second * v2[j].second;
            n1  += v1[i].second * v1[i].second;
            ++i; ++j;
        } else if (v1[i].first < v2[j].first) {
            n1 += v1[i].second * v1[i].second;
            ++i;
        } else ++j;
    }
    while (i < v1.size()) { n1 += v1[i].second * v1[i].second; ++i; }
    if (n1 == 0 || v2_norm == 0) return 0.0;
    return dot / (sqrt(n1) * sqrt(v2_norm));
}

string error_type_from_output(const string& output) {
    string lo = output;
    transform(lo.begin(), lo.end(), lo.begin(), ::tolower);
    if (lo.find("no such file") != string::npos) return "missing_file";
    if (lo.find("name or service not known") != string::npos || lo.find("name resolution") != string::npos)
        return "name_resolution";
    if (lo.find("segmentation fault") != string::npos || lo.find("return code: 139") != string::npos
        || lo.find("return code: -11") != string::npos) return "segmentation_fault";
    if (lo.find("return code: 1") != string::npos) return "execution_error";
    if (lo.find("compilation") != string::npos || lo.find("g++") != string::npos) return "compilation_error";
    return "unknown";
}

struct AgentConfig {
    string backup_dir = ".agent_backups";
    string memory_db_path = "./agent_memory_db";
    int max_execution_timeout = 30;
    int max_code_timeout = 15;
    int max_install_timeout = 60;
    double semantic_similarity_threshold = 0.72;
    int max_memory_entries = 50;
    map<string,string> models = {
        {"groq","llama-3.3-70b-versatile"},
        {"google","gemini-2.5-flash"},
        {"openrouter","meta-llama/llama-3.3-70b-instruct"}
    };
    vector<string> backend_priority = {"groq","openrouter","google"};
    int global_rpm = 30;
    int max_actions_per_turn = 6;
    json evolution = json::parse(R"({"enabled":true,"auto_reflect":true,"max_context_turns":6})");

    static AgentConfig from_env() { return AgentConfig(); }
};

class RateLimiter {
    double interval;
    double last_call = 0.0;
    mutex mtx;
public:
    RateLimiter(int rpm=60) : interval(60.0/rpm) {}
    void acquire() {
        unique_lock<mutex> lock(mtx);
        double now = get_time();
        double st = last_call + interval - now;
        if (st > 0) {
            lock.unlock();
            this_thread::sleep_for(chrono::duration<double>(st));
            lock.lock();
        }
        last_call = get_time();
    }
};

size_t curl_write_cb(void* contents, size_t size, size_t nmemb, string* userp) {
    userp->append((char*)contents, size*nmemb);
    return size*nmemb;
}

tuple<long,string> http_post(const string& url, const string& payload,
                             const vector<pair<string,string>>& headers, long timeout_sec=90) {
    CURL* curl = curl_easy_init();
    string body;
    long status = 0;
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
        struct curl_slist* hdrs = nullptr;
        for (const auto& [k,v] : headers)
            hdrs = curl_slist_append(hdrs, (k+": "+v).c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, hdrs);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &body);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout_sec);
        curl_easy_perform(curl);
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
        curl_slist_free_all(hdrs);
        curl_easy_cleanup(curl);
    }
    return {status, body};
}

//class LLMClientBase {
//protected:
    //string api_key;
    //unordered_map<string, tuple<string,bool,string>> cache;
    //vector<string> cache_keys;
    //size_t cache_max = 128;
    //mutex cache_mtx;
    //string make_cache_key(const string& model, const vector<pair<string,string>>& messages,
                          //double temperature, int max_tokens) {
        //json j = json::array();
        //j.push_back(model);
        //json msgs = json::array();
        //for (const auto& [r,c] : messages) {
            //json m = json::array(); m.push_back(r); m.push_back(c); msgs.push_back(m);
        //}
        //j.push_back(msgs);
        //j.push_back(temperature);
        //j.push_back(max_tokens);
        //return md5_hash(j.dump());
    //}
//public:
    //LLMClientBase(const string& k) : api_key(k) {}
    //virtual ~LLMClientBase() = default;
    //virtual tuple<string,bool,string> fetch(const string& model,
        //const vector<pair<string,string>>& messages, double temperature, int max_tokens) = 0;
    //tuple<string,bool,string> chat(const string& model, const vector<pair<string,string>>& messages,
        //double temperature=0.0, int max_tokens=2048, bool stream=false,
        //int max_attempts=3, double base_delay=1.0) {
        //string key = make_cache_key(model, messages, temperature, max_tokens);
        //{
            //lock_guard<mutex> lock(cache_mtx);
            //auto it = cache.find(key);
            //if (it != cache.end()) return it->second;
        //}
        //random_device rd;
        //mt19937 gen(rd());
        //uniform_real_distribution<> dis(0.5, 1.5);
        //for (int attempt=0; attempt<max_attempts; ++attempt) {
            //try {
                //auto [content, ok, err] = fetch(model, messages, temperature, max_tokens);
                //if (ok) {
                    //lock_guard<mutex> lock(cache_mtx);
                    //if (cache.find(key) == cache.end()) {
                        //if (cache.size() >= cache_max) {
                            //string old = cache_keys.front();
                            //cache_keys.erase(cache_keys.begin());
                            //cache.erase(old);
                        //}
                        //cache[key] = {content, true, ""};
                        //cache_keys.push_back(key);
                    //}
                    //return {content, true, ""};
                //}
                //if (attempt == max_attempts-1) return {"", false, err};
            //} catch (const exception& e) {
                //if (attempt == max_attempts-1) return {"", false, e.what()};
            //}
            //double sleep_time = min(60.0, base_delay * pow(2, attempt)) * dis(gen);
            //this_thread::sleep_for(chrono::duration<double>(sleep_time));
        //}
        //return {"", false, "Max retries exceeded"};
    //}
//};

class LLMClientBase {
protected:
    string api_key;
    unordered_map<string, tuple<string,bool,string>> cache;
    vector<string> cache_keys;
    size_t cache_max = 128;
    mutex cache_mtx;
    string make_cache_key(const string& model, const vector<pair<string,string>>& messages,
                          double temperature, int max_tokens, bool json_mode) {
        json j = json::array();
        j.push_back(model);
        json msgs = json::array();
        for (const auto& [r,c] : messages) {
            json m = json::array(); m.push_back(r); m.push_back(c); msgs.push_back(m);
        }
        j.push_back(msgs);
        j.push_back(temperature);
        j.push_back(max_tokens);
        j.push_back(json_mode);
        return md5_hash(j.dump());
    }
public:
    LLMClientBase(const string& k) : api_key(k) {}
    virtual ~LLMClientBase() = default;
    virtual tuple<string,bool,string> fetch(const string& model,
        const vector<pair<string,string>>& messages, double temperature, int max_tokens,
        bool json_mode = false) = 0;
    tuple<string,bool,string> chat(const string& model, const vector<pair<string,string>>& messages,
        double temperature=0.0, int max_tokens=2048, bool stream=false,
        int max_attempts=3, double base_delay=1.0, bool json_mode = false) {
        string key = make_cache_key(model, messages, temperature, max_tokens, json_mode);
        {
            lock_guard<mutex> lock(cache_mtx);
            auto it = cache.find(key);
            if (it != cache.end()) return it->second;
        }
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(0.5, 1.5);
        for (int attempt=0; attempt<max_attempts; ++attempt) {
            try {
                auto [content, ok, err] = fetch(model, messages, temperature, max_tokens, json_mode);
                if (ok) {
                    lock_guard<mutex> lock(cache_mtx);
                    if (cache.find(key) == cache.end()) {
                        if (cache.size() >= cache_max) {
                            string old = cache_keys.front();
                            cache_keys.erase(cache_keys.begin());
                            cache.erase(old);
                        }
                        cache[key] = {content, true, ""};
                        cache_keys.push_back(key);
                    }
                    return {content, true, ""};
                }
                if (attempt == max_attempts-1) return {"", false, err};
            } catch (const exception& e) {
                if (attempt == max_attempts-1) return {"", false, e.what()};
            }
            double sleep_time = min(60.0, base_delay * pow(2, attempt)) * dis(gen);
            this_thread::sleep_for(chrono::duration<double>(sleep_time));
        }
        return {"", false, "Max retries exceeded"};
    }
};

//class GroqClient : public LLMClientBase {
    //inline static const string BASE = "https://api.groq.com/openai/v1/chat/completions";
//public:
    //GroqClient(const string& k) : LLMClientBase(k) {}
    //tuple<string,bool,string> fetch(const string& model,
        //const vector<pair<string,string>>& messages, double temperature, int max_tokens) override {
        //json payload;
        //payload["model"] = model;
        //payload["temperature"] = temperature;
        //payload["max_tokens"] = max_tokens;
        //json msgs = json::array();
        //for (const auto& [r,c] : messages) msgs.push_back({{"role",r},{"content",c}});
        //payload["messages"] = msgs;
        //vector<pair<string,string>> hdrs = {
            //{"Authorization","Bearer "+api_key},
            //{"Content-Type","application/json"}
        //};
        //auto [status, body] = http_post(BASE, payload.dump(), hdrs);
        //if (status == 200) {
            //auto d = json::parse(body);
            //return {d["choices"][0]["message"]["content"], true, ""};
        //}
        //return {"", false, "HTTP "+to_string(status)+": "+body};
    //}
//};

class GroqClient : public LLMClientBase {
    inline static const string BASE = "https://api.groq.com/openai/v1/chat/completions";
public:
    GroqClient(const string& k) : LLMClientBase(k) {}
    tuple<string,bool,string> fetch(const string& model,
        const vector<pair<string,string>>& messages, double temperature, int max_tokens,
        bool json_mode = false) override {
        json payload;
        payload["model"] = model;
        payload["temperature"] = temperature;
        payload["max_tokens"] = max_tokens;
        if (json_mode) payload["response_format"] = {{"type", "json_object"}};
        json msgs = json::array();
        for (const auto& [r,c] : messages) msgs.push_back({{"role",r},{"content",c}});
        payload["messages"] = msgs;
        vector<pair<string,string>> hdrs = {
            {"Authorization","Bearer "+api_key},
            {"Content-Type","application/json"}
        };
        auto [status, body] = http_post(BASE, payload.dump(), hdrs);
        if (status == 200) {
            auto d = json::parse(body);
            return {d["choices"][0]["message"]["content"], true, ""};
        }
        return {"", false, "HTTP "+to_string(status)+": "+body};
    }
};

//class OpenRouterClient : public LLMClientBase {
    //inline static const string BASE = "https://openrouter.ai/api/v1/chat/completions";
    //RateLimiter rl;
//public:
    //OpenRouterClient(const string& k, int rpm=30) : LLMClientBase(k), rl(rpm) {}
    //tuple<string,bool,string> fetch(const string& model,
        //const vector<pair<string,string>>& messages, double temperature, int max_tokens) override {
        //rl.acquire();
        //json payload;
        //payload["model"] = model;
        //payload["temperature"] = temperature;
        //payload["max_tokens"] = max_tokens;
        //json msgs = json::array();
        //for (const auto& [r,c] : messages) msgs.push_back({{"role",r},{"content",c}});
        //payload["messages"] = msgs;
        //vector<pair<string,string>> hdrs = {
            //{"Authorization","Bearer "+api_key},
            //{"Content-Type","application/json"},
            //{"HTTP-Referer","https://localhost"},
            //{"X-Title","CodingAgent"}
        //};
        //auto [status, body] = http_post(BASE, payload.dump(), hdrs);
        //if (status == 200) {
            //auto d = json::parse(body);
            //return {d["choices"][0]["message"]["content"], true, ""};
        //}
        //return {"", false, "HTTP "+to_string(status)+": "+body};
    //}
//};

class OpenRouterClient : public LLMClientBase {
    inline static const string BASE = "https://openrouter.ai/api/v1/chat/completions";
    RateLimiter rl;
public:
    OpenRouterClient(const string& k, int rpm=30) : LLMClientBase(k), rl(rpm) {}
    tuple<string,bool,string> fetch(const string& model,
        const vector<pair<string,string>>& messages, double temperature, int max_tokens,
        bool json_mode = false) override {
        rl.acquire();
        json payload;
        payload["model"] = model;
        payload["temperature"] = temperature;
        payload["max_tokens"] = max_tokens;
        if (json_mode) payload["response_format"] = {{"type", "json_object"}};
        json msgs = json::array();
        for (const auto& [r,c] : messages) msgs.push_back({{"role",r},{"content",c}});
        payload["messages"] = msgs;
        vector<pair<string,string>> hdrs = {
            {"Authorization","Bearer "+api_key},
            {"Content-Type","application/json"},
            {"HTTP-Referer","https://localhost"},
            {"X-Title","CodingAgent"}
        };
        auto [status, body] = http_post(BASE, payload.dump(), hdrs);
        if (status == 200) {
            auto d = json::parse(body);
            return {d["choices"][0]["message"]["content"], true, ""};
        }
        return {"", false, "HTTP "+to_string(status)+": "+body};
    }
};

//class GoogleClient : public LLMClientBase {
//public:
    //GoogleClient(const string& k) : LLMClientBase(k) {}
    //tuple<string,bool,string> fetch(const string& model,
        //const vector<pair<string,string>>& messages, double temperature, int max_tokens) override {
        //string url = "https://generativelanguage.googleapis.com/v1beta/models/"
                     //+ model + ":generateContent?key=" + api_key;
        //json contents = json::array();
        //json sys_inst;
        //bool has_sys = false;
        //for (const auto& [role, text] : messages) {
            //if (role == "system") {
                //sys_inst = json{{"parts", json::array({{{"text", text}}})}};
                //has_sys = true;
            //} else {
                //string mr = (role == "user") ? "user" : "model";
                //if (!contents.empty() && contents.back()["role"] == mr) {
                    //string cur = contents.back()["parts"][0]["text"];
                    //contents.back()["parts"][0]["text"] = cur + "\n\n" + text;
                //} else {
                    //contents.push_back({{"role",mr},{"parts",{{{"text",text}}}}});
                //}
            //}
        //}
        //if (!contents.empty() && contents[0]["role"] == "model")
            //contents.insert(contents.begin(), {{"role","user"},{"parts",{{{"text","Continue your work."}}}}});
        //json payload;
        //payload["contents"] = contents;
        //payload["generationConfig"] = {{"temperature",temperature},{"maxOutputTokens",max_tokens}};
        //if (has_sys) payload["systemInstruction"] = sys_inst;
        //vector<pair<string,string>> hdrs = {{"Content-Type","application/json"}};
        //auto [status, body] = http_post(url, payload.dump(), hdrs);
        //if (status == 200) {
            //auto d = json::parse(body);
            //if (d.contains("candidates") && !d["candidates"].empty()) {
                //try {
                    //string c = d["candidates"][0]["content"]["parts"][0]["text"];
                    //return {c, true, ""};
                //} catch (...) { return {"", false, "Unexpected response: "+body}; }
            //}
            //return {"", false, "No candidates: "+body};
        //}
        //return {"", false, "HTTP "+to_string(status)+": "+body};
    //}
//};

class GoogleClient : public LLMClientBase {
public:
    GoogleClient(const string& k) : LLMClientBase(k) {}
    tuple<string,bool,string> fetch(const string& model,
        const vector<pair<string,string>>& messages, double temperature, int max_tokens,
        bool json_mode = false) override {
        string url = "https://generativelanguage.googleapis.com/v1beta/models/"
                     + model + ":generateContent?key=" + api_key;
        json contents = json::array();
        json sys_inst;
        bool has_sys = false;
        for (const auto& [role, text] : messages) {
            if (role == "system") {
                sys_inst = json{{"parts", json::array({{{"text", text}}})}};
                has_sys = true;
            } else {
                string mr = (role == "user") ? "user" : "model";
                if (!contents.empty() && contents.back()["role"] == mr) {
                    string cur = contents.back()["parts"][0]["text"];
                    contents.back()["parts"][0]["text"] = cur + "\n\n" + text;
                } else {
                    contents.push_back({{"role",mr},{"parts",{{{"text",text}}}}});
                }
            }
        }
        if (!contents.empty() && contents[0]["role"] == "model")
            contents.insert(contents.begin(), {{"role","user"},{"parts",{{{"text","Continue your work."}}}}});
        json payload;
        payload["contents"] = contents;
        json genConfig = {{"temperature",temperature},{"maxOutputTokens",max_tokens}};
        if (json_mode) genConfig["responseMimeType"] = "application/json";
        payload["generationConfig"] = genConfig;
        if (has_sys) payload["systemInstruction"] = sys_inst;
        vector<pair<string,string>> hdrs = {{"Content-Type","application/json"}};
        auto [status, body] = http_post(url, payload.dump(), hdrs);
        if (status == 200) {
            auto d = json::parse(body);
            if (d.contains("candidates") && !d["candidates"].empty()) {
                try {
                    string c = d["candidates"][0]["content"]["parts"][0]["text"];
                    return {c, true, ""};
                } catch (...) { return {"", false, "Unexpected response: "+body}; }
            }
            return {"", false, "No candidates: "+body};
        }
        return {"", false, "HTTP "+to_string(status)+": "+body};
    }
};

tuple<int,string,string> run_cmd(const vector<string>& cmd, int timeout_sec,
                                 const string& cwd=".",
                                 const optional<string>& stdin_data = nullopt) {
    int out_p[2], err_p[2];
    int stdin_p[2] = {-1, -1};
    if (pipe(out_p) == -1) {
        cerr << "Failed to create stdout pipe" << endl;
        return {-1, "", "stdout pipe creation failed"};
    }
    if (pipe(err_p) == -1) {
        cerr << "Failed to create stderr pipe" << endl;
        close(out_p[0]); close(out_p[1]);
        return {-1, "", "stderr pipe creation failed"};
    }
    if (stdin_data) {
        if (pipe(stdin_p) == -1) {
            cerr << "Failed to create stdin pipe" << endl;
            close(out_p[0]); close(out_p[1]);
            close(err_p[0]); close(err_p[1]);
            return {-1, "", "stdin pipe creation failed"};
        }
    }
    pid_t pid = fork();
    if (pid == 0) {
        close(out_p[0]);
        close(err_p[0]);
        if (stdin_data && stdin_p[0] != -1) {
            close(stdin_p[1]);
            dup2(stdin_p[0], STDIN_FILENO);
            close(stdin_p[0]);
        }
        dup2(out_p[1], STDOUT_FILENO);
        dup2(err_p[1], STDERR_FILENO);
        close(out_p[1]);
        close(err_p[1]);
        vector<char*> argv;
        for (auto& s : cmd) argv.push_back(const_cast<char*>(s.c_str()));
        argv.push_back(nullptr);
        execvp(argv[0], argv.data());
        _exit(127);
    }
    close(out_p[1]);
    close(err_p[1]);
    if (stdin_p[0] != -1) close(stdin_p[0]);
    atomic<bool> done{false};
    string out_str, err_str;
    thread reader([&]() {
        char buf[4096];
        ssize_t n;
        while ((n = read(out_p[0], buf, sizeof(buf)-1)) > 0) { buf[n]=0; out_str += buf; }
        while ((n = read(err_p[0], buf, sizeof(buf)-1)) > 0) { buf[n]=0; err_str += buf; }
        done = true;
    });
    if (stdin_data && stdin_p[1] != -1) {
        const string& data = *stdin_data;
        ssize_t written = 0;
        ssize_t total = data.size();
        while (written < total) {
            ssize_t n = write(stdin_p[1], data.data() + written, total - written);
            if (n <= 0) break;
            written += n;
        }
        close(stdin_p[1]);
    }
    bool timed_out = false;
    auto start = chrono::steady_clock::now();
    while (!done) {
        auto elapsed = chrono::duration_cast<chrono::seconds>(
            chrono::steady_clock::now() - start).count();
        if (elapsed >= timeout_sec) { timed_out = true; kill(pid, SIGKILL); break; }
        this_thread::sleep_for(chrono::milliseconds(10));
    }
    reader.join();
    close(out_p[0]);
    close(err_p[0]);
    if (timed_out) return {-9, out_str, err_str};
    int status;
    waitpid(pid, &status, 0);
    int rc = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    return {rc, out_str, err_str};
}

class CodeExecutor {
public:
    AgentConfig config;
    fs::path log_path = "./execution_traceback.log";
    string last_compiled_exe;
    CodeExecutor(const AgentConfig& c) : config(c) {}
    tuple<int,string,string> _run(const vector<string>& cmd, int timeout,
                                  const string& cwd=".",
                                  const optional<string>& stdin_data = nullopt) {
        return run_cmd(cmd, timeout, cwd, stdin_data);
    }
    string run_shell_command(const string& command, int timeout=30) {
        auto [rc, out, err] = _run({"/bin/sh","-c",command}, timeout);
        return "Return code: " + to_string(rc) + "\nSTDOUT:\n" + out + "\nSTDERR:\n" + err;
    }
    string execute_file(const string& file_path, const optional<string>& stdin_data=nullopt) {
        fs::path path(file_path);
        string ext = path.extension().string();
        transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        vector<string> cmd;
        if (ext.empty() && !last_compiled_exe.empty() && fs::exists(last_compiled_exe)) {
            cmd = {last_compiled_exe};
        } else if (ext == ".py") {
            cmd = {"python3", path.string()};
        } else if (ext == ".cpp" || ext == ".cc" || ext == ".cxx") {
            string exe = compile_cpp(path.string());
            if (exe.find("COMPILATION_FAILED:") != string::npos) return exe;
            cmd = {exe};
        } else {
            if (fs::exists(path) && access(path.c_str(), X_OK) == 0)
                cmd = {fs::canonical(path).string()};
            else
                return "Unsupported file type: " + ext;
        }
        try {
            auto [rc, out, err] = _run(cmd, config.max_execution_timeout, ".", stdin_data);
            if (rc != 0) {
                ofstream lf(log_path, ios::app);
                lf << "--- FAILURE " << file_path << " RC=" << rc << " ---\nSTDERR:\n" << err << "\n";
                if ((ext == ".cpp" || ext == ".cc" || ext == ".cxx") && !last_compiled_exe.empty()) {
                    try { fs::remove(last_compiled_exe); } catch (...) {}
                }
                return "Return code: " + to_string(rc) + "\nSTDOUT:\n" + out + "\nSTDERR:\n" + err;
            }
            return "Return code: 0\nSTDOUT:\n" + out + "\nSTDERR:\n" + err;
        } catch (const exception& e) {
            return "Error: " + string(e.what());
        }
    }
    string execute_code(const string& code, const string& language="auto") {
        string lang = language;
        if (lang == "auto")
            lang = (code.find("#include") != string::npos || code.find("int main(") != string::npos) ? "cpp" : "python";
        string tmp = "/tmp/tmp_" + to_string(getpid());
        if (lang == "cpp") {
            tmp += ".cpp";
            ofstream f(tmp); f << code; f.close();
            string r = execute_file(tmp);
            fs::remove(tmp);
            return r;
        } else if (lang == "python") {
            string syn = validate_syntax(code, "python");
            if (syn.find("FAILED") != string::npos) return syn + "\n(Execution skipped)";
            tmp += ".py";
            ofstream f(tmp); f << code; f.close();
            string r = execute_file(tmp);
            fs::remove(tmp);
            return r;
        }
        return "Unsupported language: " + lang;
    }
    string compile_cpp(const string& source_file, const string& output_file="") {
        string out = output_file;
        if (out.empty()) out = "/tmp/" + fs::path(source_file).stem().string() + "_" + to_string(getpid()) + ".out";
        vector<string> flags = {"-std=c++17","-O2"};
        ifstream sf(source_file);
        string content((istreambuf_iterator<char>(sf)), istreambuf_iterator<char>());
        if (!sf) return "COMPILATION_FAILED: cannot read source";
        if (content.find("std::thread") != string::npos) flags.push_back("-pthread");
        if (content.find("<curl/curl.h>") != string::npos) flags.push_back("-lcurl");
        if (content.find("<openssl/") != string::npos) { flags.push_back("-lcrypto"); flags.push_back("-lssl"); }
        if (content.find("<nlohmann/json.hpp>") != string::npos) flags.push_back("-I/usr/include/nlohmann");
        vector<string> cmd = {"g++"};
        cmd.insert(cmd.end(), flags.begin(), flags.end());
        cmd.push_back(source_file); cmd.push_back("-o"); cmd.push_back(out);
        auto [rc, _, err] = _run(cmd, 15);
        if (rc != 0) return "COMPILATION_FAILED: " + err;
        last_compiled_exe = out;
        return out;
    }
    string validate_syntax(const string& code, const string& language="auto") {
        string lang = language;
        if (lang == "auto")
            lang = (code.find("#include") != string::npos || code.find("int main(") != string::npos) ? "cpp" : "python";
        string tmp = "/tmp/tmp_" + to_string(getpid());
        if (lang == "cpp") {
            tmp += ".cpp";
            ofstream f(tmp); f << code; f.close();
            auto [rc, _, err] = _run({"g++","-std=c++17","-fsyntax-only",tmp}, 10);
            fs::remove(tmp);
            return rc == 0 ? "Syntax validation PASSED." : "Syntax validation FAILED: " + err;
        } else {
            tmp += ".py";
            ofstream f(tmp); f << code; f.close();
            auto [rc, _, err] = _run({"python3","-m","py_compile",tmp}, 10);
            fs::remove(tmp);
            return rc == 0 ? "Syntax validation PASSED." : "Syntax validation FAILED: " + err;
        }
    }
};

struct FailurePattern {
    string task_hash;
    string error_type;
    string error_signature;
    json fix_actions;
    string hint;
    int occurrence_count = 1;
    double last_seen = get_time();
};

class SemanticMemoryEntry {
public:
    string task_hash;
    string task_text;
    string solution_code;
    vector<pair<string,double>> word_vector;
    double norm = 0.0;
    double success_timestamp;
    int api_calls_saved = 0;
    int access_count = 0;
    double last_access_time = 0.0;
    unordered_set<string> term_set;
    int term_count = 0;
    bool is_plan = false;
    SemanticMemoryEntry(const string& th, const string& tt, const string& sc)
        : task_hash(th), task_text(tt), solution_code(sc), success_timestamp(get_time()) {}
};

class SemanticMemoryManager {
    AgentConfig config;
    list<SemanticMemoryEntry> memory;
    unordered_map<string, SemanticMemoryEntry*> exact;
    unordered_map<string, vector<SemanticMemoryEntry*>> inverted;
    unordered_map<string,double> idf;
    int total_docs = 0;
    unordered_map<string, float> stats = {
        {"exact_hits",0},{"semantic_hits",0},{"misses",0},
        {"false_positive_avoided",0},{"total_retrieval_time_ms",0.0f},{"retrieval_count",0}
    };
    unordered_map<string,double> dynamic_thresholds;
    int max_entries;
    fs::path db_path;
    vector<FailurePattern> failure_patterns;
    unordered_map<string, vector<FailurePattern*>> error_index;
    unordered_map<string, tuple<double,double,string>> last_retrieval;
    vector<double> weights = {0.6, 0.4, 0.0};
    double learning_rate = 0.02;
    unordered_map<string,int> consec_fail;
public:
    SemanticMemoryManager(const AgentConfig& c) : config(c), max_entries(c.max_memory_entries),
        db_path(c.memory_db_path) {
        fs::create_directories(db_path);
        load_all();
    }
    void load_all() { load_memory(); load_failures(); }
    void load_memory() {
        fs::path f = db_path / "semantic_memory.json";
        if (!fs::exists(f)) return;
        try {
            ifstream file(f);
            json data; file >> data;
            for (const auto& item : data) {
                memory.emplace_back(item["task_hash"], item["task_text"], item["solution_code"]);
                auto& e = memory.back();
                e.success_timestamp = item.value("timestamp", 0.0);
                e.api_calls_saved = item.value("api_calls_saved", 0);
                e.access_count = item.value("access_count", 0);
                e.last_access_time = item.value("last_access_time", 0.0);
                if (item.contains("vector")) {
                    for (auto& [k,v] : item["vector"].items())
                        e.word_vector.emplace_back(k, v.get<double>());
                }
                if (item.contains("terms")) {
                    for (const auto& t : item["terms"]) e.term_set.insert(t.get<string>());
                }
                e.term_count = e.term_set.size();
                e.norm = item.value("norm", 0.0);
                e.is_plan = item.value("is_plan", false);
                exact[e.task_hash] = &e;
                for (const auto& term : e.term_set) inverted[term].push_back(&e);
            }
            update_idf();
        } catch (const exception& e) { cout << "[Memory] Load error: " << e.what() << endl; }
    }
    void load_failures() {
        fs::path f = db_path / "failure_patterns.json";
        if (!fs::exists(f)) return;
        try {
            ifstream file(f);
            json data; file >> data;
            for (const auto& item : data) {
                FailurePattern p;
                p.task_hash = item["task_hash"];
                p.error_type = item["error_type"];
                p.error_signature = item["error_signature"];
                p.fix_actions = item["fix_actions"];
                p.hint = item["hint"];
                p.occurrence_count = item.value("occurrence_count", 1);
                p.last_seen = item.value("last_seen", 0.0);
                failure_patterns.push_back(p);
                error_index[p.error_type].push_back(&failure_patterns.back());
            }
        } catch (const exception& e) { cout << "[FailureMemory] Load error: " << e.what() << endl; }
    }
    void save_all() { save_memory(); save_failures(); }
    void save_memory() {
        try {
            json data = json::array();
            for (auto it = memory.rbegin(); it != memory.rend(); ++it) {
                auto& e = *it;
                json item;
                item["task_hash"] = e.task_hash;
                item["task_text"] = e.task_text.substr(0, 200);
                item["solution_code"] = e.solution_code;
                item["timestamp"] = e.success_timestamp;
                item["api_calls_saved"] = e.api_calls_saved;
                item["access_count"] = e.access_count;
                item["last_access_time"] = e.last_access_time;
                json vec;
                for (const auto& [k,v] : e.word_vector) vec[k] = v;
                item["vector"] = vec;
                item["terms"] = e.term_set;
                item["norm"] = e.norm;
                item["is_plan"] = e.is_plan;
                data.push_back(item);
            }
            ofstream f(db_path / "semantic_memory.json");
            f << data.dump(2);
        } catch (const exception& e) { cout << "[Memory] Save error: " << e.what() << endl; }
    }
    void save_failures() {
        try {
            json data = json::array();
            for (auto& p : failure_patterns) {
                json item;
                item["task_hash"] = p.task_hash;
                item["error_type"] = p.error_type;
                item["error_signature"] = p.error_signature;
                item["fix_actions"] = p.fix_actions;
                item["hint"] = p.hint;
                item["occurrence_count"] = p.occurrence_count;
                item["last_seen"] = p.last_seen;
                data.push_back(item);
            }
            ofstream f(db_path / "failure_patterns.json");
            f << data.dump(2);
        } catch (const exception& e) { cout << "[FailureMemory] Save error: " << e.what() << endl; }
    }
    void update_idf() {
        total_docs = memory.size();
        unordered_map<string,int> df;
        for (const auto& e : memory)
            for (const auto& term : e.term_set) df[term]++;
        idf.clear();
        for (const auto& [term, f] : df)
            idf[term] = log(1.0 + total_docs / (double)f);
    }
    vector<pair<string,double>> compute_vector(const string& text) {
        return compute_tfidf_vector(text, idf, total_docs);
    }
    optional<string> retrieve_similar(const string& query, double threshold=0.75) {
        auto start = chrono::steady_clock::now();
        string task_hash = md5_hash(query);
        auto it = exact.find(task_hash);
        if (it != exact.end()) {
            auto* entry = it->second;
            entry->access_count++;
            entry->last_access_time = get_time();
            entry->api_calls_saved++;
            save_all();
            stats["exact_hits"]++;
            stats["retrieval_count"]++;
            auto elapsed = chrono::duration_cast<chrono::microseconds>(
                chrono::steady_clock::now() - start).count() / 1000.0;
            stats["total_retrieval_time_ms"] += elapsed;
            return entry->solution_code;
        }
        if (memory.empty()) {
            stats["misses"]++;
            return nullopt;
        }
        auto qvec = compute_vector(query);
        if (qvec.empty()) {
            stats["misses"]++;
            return nullopt;
        }
        auto cands = get_candidates(qvec);
        double best_score = 0.0, second_score = 0.0;
        SemanticMemoryEntry* best_entry = nullptr;
        double cosv = 0.0, jacv = 0.0;
        for (auto* entry : cands) {
            double cos = cosine_similarity(qvec, entry->word_vector, entry->norm);
            unordered_set<string> qterms;
            for (const auto& p : qvec) qterms.insert(p.first);
            double jac = jaccard(qterms, entry->term_set);
            double score = hybrid(cos, jac);
            double age_days = (get_time() - entry->success_timestamp) / 86400.0;
            score *= (1.0 - 0.1 * min(1.0, max(0.0, (age_days - 1.0) / 6.0)));
            if (score > best_score) {
                second_score = best_score;
                best_score = score;
                best_entry = entry;
                cosv = cos; jacv = jac;
            } else if (score > second_score) {
                second_score = score;
            }
        }
        double eff_thresh = threshold;
        auto dt = dynamic_thresholds.find(task_hash);
        if (dt != dynamic_thresholds.end()) eff_thresh = dt->second;
        if (best_score >= eff_thresh && (second_score == 0 || best_score / second_score >= 1.15)) {
            stats["semantic_hits"]++;
            stats["retrieval_count"]++;
            auto elapsed = chrono::duration_cast<chrono::microseconds>(
                chrono::steady_clock::now() - start).count() / 1000.0;
            stats["total_retrieval_time_ms"] += elapsed;
            best_entry->access_count++;
            best_entry->last_access_time = get_time();
            best_entry->api_calls_saved++;
            save_all();
            last_retrieval[task_hash] = {cosv, jacv, best_entry->task_hash};
            return best_entry->solution_code;
        }
        double base_threshold = config.semantic_similarity_threshold;
        double adaptive_min = 0.3 + 0.2 * (1.0 - exp(-0.1 * memory.size()));
        if (best_entry && best_score >= adaptive_min &&
            (best_score >= base_threshold * 0.7 || memory.size() > 10)) {
            best_entry->access_count++;
            best_entry->last_access_time = get_time();
            best_entry->api_calls_saved++;
            save_all();
            stats["semantic_hits"]++;
            stats["retrieval_count"]++;
            auto elapsed = chrono::duration_cast<chrono::microseconds>(
                chrono::steady_clock::now() - start).count() / 1000.0;
            stats["total_retrieval_time_ms"] += elapsed;
            last_retrieval[task_hash] = {cosv, jacv, best_entry->task_hash};
            return best_entry->solution_code;
        }
        stats["misses"]++;
        return nullopt;
    }
    double jaccard(const unordered_set<string>& s1, const unordered_set<string>& s2) {
        if (s1.empty() || s2.empty()) return 0.0;
        int inter = 0;
        for (const auto& x : s1) if (s2.count(x)) inter++;
        int uni = s1.size() + s2.size() - inter;
        return uni ? (double)inter / uni : 0.0;
    }
    double hybrid(double cos, double jac) {
        double z = weights[0]*cos + weights[1]*jac + weights[2];
        if (z >= 0) return 1.0 / (1.0 + exp(-z));
        return exp(z) / (1.0 + exp(z));
    }
    vector<SemanticMemoryEntry*> get_candidates(const vector<pair<string,double>>& qvec) {
        unordered_map<string, SemanticMemoryEntry*> cand;
        for (const auto& [term, _] : qvec) {
            auto it = inverted.find(term);
            if (it != inverted.end())
                for (auto* e : it->second) cand[e->task_hash] = e;
        }
        vector<SemanticMemoryEntry*> res;
        for (const auto& [_, e] : cand) res.push_back(e);
        if (res.empty())
            for (auto& e : memory) res.push_back(&e);
        return res;
    }
    void store(const string& task, const string& solution) {
        string task_hash = md5_hash(task);
        bool is_plan = false;
        try {
            auto parsed = json::parse(solution);
            if (parsed.is_array()) {
                is_plan = true;
                for (const auto& a : parsed) if (!a.contains("tool")) { is_plan = false; break; }
            }
        } catch (...) {}
        SemanticMemoryEntry* entry = nullptr;
        auto it = exact.find(task_hash);
        if (it != exact.end()) {
            entry = it->second;
            for (const auto& term : entry->term_set) {
                auto& lst = inverted[term];
                lst.erase(remove_if(lst.begin(), lst.end(),
                    [&](auto* e){ return e->task_hash == task_hash; }), lst.end());
            }
            entry->solution_code = solution;
        } else {
            memory.emplace_back(task_hash, task, solution);
            entry = &memory.back();
            exact[task_hash] = entry;
        }
        entry->word_vector = compute_vector(task);
        entry->term_set.clear();
        for (const auto& p : entry->word_vector) entry->term_set.insert(p.first);
        entry->term_count = entry->term_set.size();
        entry->success_timestamp = get_time();
        entry->is_plan = is_plan;
        entry->access_count++;
        for (const auto& term : entry->term_set) inverted[term].push_back(entry);
        if ((int)memory.size() > max_entries) {
            auto min_it = min_element(memory.begin(), memory.end(),
                [](auto& a, auto& b){ return a.access_count < b.access_count; });
            auto* ev = &(*min_it);
            exact.erase(ev->task_hash);
            for (const auto& t : ev->term_set) {
                auto& lst = inverted[t];
                lst.erase(remove_if(lst.begin(), lst.end(),
                    [&](auto* e){ return e->task_hash == ev->task_hash; }), lst.end());
                if (lst.empty()) inverted.erase(t);
            }
            memory.erase(min_it);
        }
        update_idf();
        save_all();
    }
    void store_failure(const string& task_hash, const string& error_type,
                       const string& error_output, const json& fix_actions, const string& hint) {
        string sig = error_output.substr(0, 200);
        replace(sig.begin(), sig.end(), '\n', ' ');
        sig = trim(sig);
        auto it = error_index.find(error_type);
        if (it != error_index.end()) {
            for (auto* p : it->second) {
                if (p->error_signature.find(sig) != string::npos || sig.find(p->error_signature) != string::npos) {
                    p->occurrence_count++;
                    p->last_seen = get_time();
                    if (hint.size() > p->hint.size()) p->hint = hint;
                    save_all();
                    return;
                }
            }
        }
        FailurePattern p;
        p.task_hash = task_hash; p.error_type = error_type; p.error_signature = sig;
        p.fix_actions = fix_actions; p.hint = hint;
        failure_patterns.push_back(p);
        error_index[error_type].push_back(&failure_patterns.back());
        save_all();
    }
    string get_hint_for_error(const string& task_hash, const string& error_output) {
        string etype = error_type_from_output(error_output);
        string sig = error_output.substr(0, 200);
        replace(sig.begin(), sig.end(), '\n', ' ');
        sig = trim(sig);
        auto it = error_index.find(etype);
        if (it != error_index.end()) {
            for (auto* p : it->second) {
                if (p->error_signature.find(sig) != string::npos || sig.find(p->error_signature) != string::npos) {
                    p->occurrence_count++;
                    p->last_seen = get_time();
                    save_all();
                    return p->hint;
                }
            }
        }
        return "";
    }
    void report_outcome(const string& task_hash, bool success) {
        auto it = last_retrieval.find(task_hash);
        if (it != last_retrieval.end()) {
            auto [cos, jac, entry_hash] = it->second;
            last_retrieval.erase(it);
            double target = success ? 1.0 : 0.0;
            double z = weights[0]*cos + weights[1]*jac + weights[2];
            double pred = 1.0 / (1.0 + exp(-z));
            double grad = pred - target;
            weights[0] -= learning_rate * grad * cos;
            weights[1] -= learning_rate * grad * jac;
            weights[2] -= learning_rate * grad;
            for (int i = 0; i < 3; ++i)
                weights[i] = max(-5.0, min(5.0, weights[i]));
        }
        double thresh = config.semantic_similarity_threshold;
        if (success) {
            thresh = max(0.50, thresh * 0.95);
        } else {
            consec_fail[task_hash]++;
            if (consec_fail[task_hash] >= 2) {
                thresh = min(0.90, thresh * 1.05);
                consec_fail[task_hash] = 0;
            }
        }
        dynamic_thresholds[task_hash] = thresh;
    }
    void print_stats() {
        int saved = 0;
        for (const auto& e : memory) saved += e.api_calls_saved;
        cout << "Memory entries: " << memory.size() << "/" << max_entries << endl;
        cout << "Hits exact=" << stats["exact_hits"] << " semantic=" << stats["semantic_hits"]
             << " misses=" << stats["misses"] << endl;
        cout << "API calls saved: " << saved << endl;
    }
};

class ReflectionEngine {
    unordered_map<string,int> patterns;
    vector<json> failure_log;
public:
    json analyze(const string& task, bool success, const string& observations, int turn_count) {
        json analysis;
        analysis["timestamp"] = get_time();
        analysis["task_hash"] = md5_hash(task);
        analysis["success"] = success;
        analysis["turn_count"] = turn_count;
        analysis["error_type"] = nullptr;
        analysis["bottleneck"] = nullptr;
        analysis["suggested_capability"] = nullptr;
        if (!success) {
            string etype = error_type_from_output(observations);
            analysis["error_type"] = etype;
            if (etype == "segmentation_fault") {
                analysis["bottleneck"] = "null_pointer_or_buffer_overflow";
                analysis["suggested_capability"] = "memory_safety_checker";
            } else if (etype == "execution_error") {
                analysis["suggested_capability"] = "execution_fixer";
            }
            patterns[etype]++;
            failure_log.push_back(analysis);
        }
        return analysis;
    }
};

//class LLMBackendManager {
    //AgentConfig config;
    //unordered_map<string, unique_ptr<LLMClientBase>> clients;
    //string current_backend;
    //unordered_map<string,double> cooldown_until;
    //RateLimiter rate_limiter;
    //atomic<bool> abort_flag{false};
    //void init_clients() {
        //const char* groq = getenv("GROQ_API_KEY");
        //const char* or_key = getenv("OPENROUTER_API_KEY");
        //const char* ggl = getenv("GOOGLE_API_KEY");
        //if (groq) clients["groq"] = make_unique<GroqClient>(groq);
        //if (or_key) clients["openrouter"] = make_unique<OpenRouterClient>(or_key);
        //if (ggl) clients["google"] = make_unique<GoogleClient>(ggl);
        //for (const auto& b : config.backend_priority) {
            //if (clients.count(b)) { current_backend = b; break; }
        //}
        //if (current_backend.empty()) throw runtime_error("No LLM backend available");
    //}
    //bool is_on_cooldown(const string& backend) {
        //auto it = cooldown_until.find(backend);
        //if (it == cooldown_until.end()) return false;
        //if (get_time() >= it->second) { cooldown_until.erase(it); return false; }
        //return true;
    //}
    //void set_cooldown(const string& backend, const string& err) {
        //string lo = err;
        //transform(lo.begin(), lo.end(), lo.begin(), ::tolower);
        //double now = get_time();
        //if (lo.find("quota") != string::npos && lo.find("tpd") != string::npos) {
            //regex re(R"(reset at (\d+))");
            //smatch m;
            //if (regex_search(lo, m, re))
                //cooldown_until[backend] = now + min((double)max(0, stoi(m[1]) - (int)now), 30.0);
            //else
                //cooldown_until[backend] = now + 30;
        //} else if (lo.find("429") != string::npos) {
            //regex re(R"(reset at (\d+))");
            //smatch m;
            //if (regex_search(lo, m, re))
                //cooldown_until[backend] = now + min((double)max(0, stoi(m[1]) - (int)now), 10.0);
            //else
                //cooldown_until[backend] = now + 10;
        //} else {
            //cooldown_until[backend] = now + 2;
        //}
    //}
//public:
    //LLMBackendManager(const AgentConfig& c) : config(c), rate_limiter(c.global_rpm) { init_clients(); }
    //void abort() { abort_flag = true; }
    //void reset_abort() { abort_flag = false; }
    //void reset_cooldowns() { cooldown_until.clear(); }
    //tuple<string,bool,string> chat(const vector<pair<string,string>>& messages,
        //double temperature=0.0, int max_tokens=2048) {
        //if (abort_flag) return {"", false, "Aborted by agent."};
        //rate_limiter.acquire();
        //if (abort_flag) return {"", false, "Aborted by agent."};
        //vector<string> order = {current_backend};
        //for (const auto& b : config.backend_priority)
            //if (b != current_backend) order.push_back(b);
        //for (int retry = 0; retry < 3; ++retry) {
            //for (const auto& backend : order) {
                //if (abort_flag) return {"", false, "Aborted by agent."};
                //if (is_on_cooldown(backend)) continue;
                //auto it = clients.find(backend);
                //if (it == clients.end()) continue;
                //string model = config.models.count(backend) ? config.models[backend] : "unknown";
                //auto [content, ok, err] = it->second->chat(model, messages, temperature, max_tokens);
                //if (ok) {
                    //cooldown_until.clear();
                    //if (backend != current_backend) {
                        //cout << "[BACKEND] Switched to healthy backend: " << backend << endl;
                        //current_backend = backend;
                    //}
                    //return {content, true, ""};
                //}
                //set_cooldown(backend, err);
            //}
            //if (retry < 2) {
                //int sleep_time = min(15, 3 * (1 << retry));
                //cout << "[BACKEND] All backends exhausted, waiting " << sleep_time
                     //<< "s (retry " << retry+1 << "/2)..." << endl;
                //this_thread::sleep_for(chrono::seconds(sleep_time));
            //} else break;
        //}
        //return {"", false, "All backends exhausted."};
    //}
    //tuple<json,bool,string> structured_chat(const vector<pair<string,string>>& messages,
        //const json& schema, double temperature=0.0, int max_tokens=2048) {
        //string sys = "You are a precise JSON-only assistant. Output ONLY valid JSON matching the schema below.\nSchema:\n"
                     //+ schema.dump(2) + "\nOutput the raw JSON immediately.";
        //vector<pair<string,string>> full = {{"system", sys}};
        //full.insert(full.end(), messages.begin(), messages.end());
        //auto [content, ok, err] = chat(full, temperature, max_tokens);
        //if (!ok) return {json{}, false, err};
        //regex code_block(R"(```(?:json)?\s*(.*?)```)", regex::icase | regex::ECMAScript);
        //smatch m;
        //if (regex_search(content, m, code_block)) {
            //content = m[1];
        //} else {
            //regex json_obj(R"(\{[\s\S]*?\}|\[[\s\S]*?\])");
            //if (regex_search(content, m, json_obj)) content = m[1];
        //}
        //content = trim(content);
        //try {
            //return {json::parse(content), true, ""};
        //} catch (const exception& e) {
            //return {json{}, false, string("JSON parse error: ") + e.what() + " (snippet: " + content.substr(0,200) + "...)"};
        //}
    //}
    //string get_current_backend() const { return current_backend.empty() ? "unknown" : current_backend; }
//};

class LLMBackendManager {
    AgentConfig config;
    unordered_map<string, unique_ptr<LLMClientBase>> clients;
    string current_backend;
    unordered_map<string,double> cooldown_until;
    RateLimiter rate_limiter;
    atomic<bool> abort_flag{false};
    void init_clients() {
        const char* groq = getenv("GROQ_API_KEY");
        const char* or_key = getenv("OPENROUTER_API_KEY");
        const char* ggl = getenv("GOOGLE_API_KEY");
        if (groq) clients["groq"] = make_unique<GroqClient>(groq);
        if (or_key) clients["openrouter"] = make_unique<OpenRouterClient>(or_key);
        if (ggl) clients["google"] = make_unique<GoogleClient>(ggl);
        for (const auto& b : config.backend_priority) {
            if (clients.count(b)) { current_backend = b; break; }
        }
        if (current_backend.empty()) throw runtime_error("No LLM backend available");
    }
    bool is_on_cooldown(const string& backend) {
        auto it = cooldown_until.find(backend);
        if (it == cooldown_until.end()) return false;
        if (get_time() >= it->second) { cooldown_until.erase(it); return false; }
        return true;
    }
    void set_cooldown(const string& backend, const string& err) {
        string lo = err;
        transform(lo.begin(), lo.end(), lo.begin(), ::tolower);
        double now = get_time();
        if (lo.find("quota") != string::npos && lo.find("tpd") != string::npos) {
            regex re(R"(reset at (\d+))");
            smatch m;
            if (regex_search(lo, m, re))
                cooldown_until[backend] = now + min((double)max(0, stoi(m[1]) - (int)now), 30.0);
            else
                cooldown_until[backend] = now + 30;
        } else if (lo.find("429") != string::npos) {
            regex re(R"(reset at (\d+))");
            smatch m;
            if (regex_search(lo, m, re))
                cooldown_until[backend] = now + min((double)max(0, stoi(m[1]) - (int)now), 10.0);
            else
                cooldown_until[backend] = now + 10;
        } else {
            cooldown_until[backend] = now + 2;
        }
    }
public:
    LLMBackendManager(const AgentConfig& c) : config(c), rate_limiter(c.global_rpm) { init_clients(); }
    void abort() { abort_flag = true; }
    void reset_abort() { abort_flag = false; }
    void reset_cooldowns() { cooldown_until.clear(); }
    tuple<string,bool,string> chat(const vector<pair<string,string>>& messages,
        double temperature=0.0, int max_tokens=2048, bool json_mode = false) {
        if (abort_flag) return {"", false, "Aborted by agent."};
        rate_limiter.acquire();
        if (abort_flag) return {"", false, "Aborted by agent."};
        vector<string> order = {current_backend};
        for (const auto& b : config.backend_priority)
            if (b != current_backend) order.push_back(b);
        for (int retry = 0; retry < 3; ++retry) {
            for (const auto& backend : order) {
                if (abort_flag) return {"", false, "Aborted by agent."};
                if (is_on_cooldown(backend)) continue;
                auto it = clients.find(backend);
                if (it == clients.end()) continue;
                string model = config.models.count(backend) ? config.models[backend] : "unknown";
                auto [content, ok, err] = it->second->chat(model, messages, temperature, max_tokens,
                                                           false, 3, 1.0, json_mode);
                if (ok) {
                    cooldown_until.clear();
                    if (backend != current_backend) {
                        cout << "[BACKEND] Switched to healthy backend: " << backend << endl;
                        current_backend = backend;
                    }
                    return {content, true, ""};
                }
                set_cooldown(backend, err);
            }
            if (retry < 2) {
                int sleep_time = min(15, 3 * (1 << retry));
                cout << "[BACKEND] All backends exhausted, waiting " << sleep_time
                     << "s (retry " << retry+1 << "/2)..." << endl;
                this_thread::sleep_for(chrono::seconds(sleep_time));
            } else break;
        }
        return {"", false, "All backends exhausted."};
    }
    tuple<json,bool,string> structured_chat(const vector<pair<string,string>>& messages,
        const json& schema, double temperature=0.0, int max_tokens=2048) {
        string sys = "You are a precise JSON-only assistant. Output ONLY valid JSON matching the schema below.\nSchema:\n"
                     + schema.dump(2) + "\nOutput the raw JSON immediately.";
        vector<pair<string,string>> full = {{"system", sys}};
        full.insert(full.end(), messages.begin(), messages.end());
        auto [content, ok, err] = chat(full, temperature, max_tokens, true);
        if (!ok) return {json{}, false, err};
        regex code_block(R"(```(?:json)?\s*(.*?)```)", regex::icase | regex::ECMAScript);
        smatch m;
        if (regex_search(content, m, code_block)) {
            content = m[1];
        } else {
            regex json_obj(R"(\{[\s\S]*?\}|\[[\s\S]*?\])");
            if (regex_search(content, m, json_obj)) content = m[1];
        }
        content = trim(content);
        try {
            return {json::parse(content), true, ""};
        } catch (const exception& e) {
            return {json{}, false, string("JSON parse error: ") + e.what() + " (snippet: " + content.substr(0,200) + "...)"};
        }
    }
    string get_current_backend() const { return current_backend.empty() ? "unknown" : current_backend; }
};

class FileManager {
    fs::path backup_dir;
public:
    FileManager(const string& dir) : backup_dir(dir) { fs::create_directories(backup_dir); }
    string backup_file(const string& file_path) {
        fs::path path(file_path);
        if (!fs::exists(path)) return "";
        regex pattern(path.filename().string() + R"(_v(\d+)\.bak)");
        int max_ver = 0;
        for (const auto& entry : fs::directory_iterator(backup_dir)) {
            smatch m;
            string name = entry.path().filename().string();
            if (regex_match(name, m, pattern)) max_ver = max(max_ver, stoi(m[1]));
        }
        fs::path backup = backup_dir / (path.filename().string() + "_v" + to_string(max_ver+1) + ".bak");
        fs::copy_file(path, backup);
        return backup.string();
    }
    string fix_incorrect_includes(const string& content) {
        string r = content;
        regex p1(R"(#include\s*<json/json\.h>)");
        regex p2(R"(#include\s*"json/json\.h")");
        r = regex_replace(r, p1, "#include <nlohmann/json.hpp>");
        r = regex_replace(r, p2, "#include <nlohmann/json.hpp>");
        return r;
    }
    string inject_missing_includes(const string& content) {
        static const vector<pair<regex,string>> needed = {
            {regex(R"(\bstd::cout\b)"), "<iostream>"},
            {regex(R"(\bstd::string\b)"), "<string>"},
            {regex(R"(\bstd::vector\b)"), "<vector>"},
            {regex(R"(\bstd::thread\b)"), "<thread>"},
            {regex(R"(\bstd::mutex\b)"), "<mutex>"},
            {regex(R"(\bstd::ofstream\b)"), "<fstream>"},
            {regex(R"(\bstd::ifstream\b)"), "<fstream>"},
            {regex(R"(\bnlohmann::json\b)"), "<nlohmann/json.hpp>"},
        };
        set<string> existing;
        regex inc_regex(R"(#include\s*[<"]([^>"]+)[>"])");
        for (sregex_iterator it(content.begin(), content.end(), inc_regex), end; it != end; ++it)
            existing.insert(it->str(1));
        set<string> to_add;
        for (const auto& [pat, inc] : needed)
            if (regex_search(content, pat) && !existing.count(inc))
                to_add.insert("#include " + inc);
        if (to_add.empty()) return content;
        vector<string> lines;
        stringstream ss(content);
        string line;
        while (getline(ss, line)) lines.push_back(line);
        int last_inc = -1;
        for (int i = 0; i < (int)lines.size(); ++i)
            if (lines[i].find("#include") == 0) last_inc = i;
        int pos = last_inc >= 0 ? last_inc + 1 : 0;
        for (const auto& inc : to_add) {
            lines.insert(lines.begin() + pos, inc);
            pos++;
        }
        string result;
        for (size_t i = 0; i < lines.size(); ++i) {
            if (i > 0) result += "\n";
            result += lines[i];
        }
        return result;
    }
    string write_file(const string& file_path, const string& content) {
        fs::path path(file_path);
        fs::path parent = path.parent_path();
        if (!parent.empty() && parent != ".") {
            fs::create_directories(parent);
        }
        string fc = content;
        string ext = path.extension().string();
        transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".cpp" || ext == ".cc" || ext == ".cxx") {
            fc = fix_incorrect_includes(fc);
            fc = inject_missing_includes(fc);
        }
        fs::path tmp = path; tmp += ".tmp";
        ofstream f(tmp);
        f << fc;
        f.close();
        fs::rename(tmp, path);
        return "File written successfully: " + file_path;
    }
    string read_file(const string& file_path) {
        try {
            ifstream f(file_path);
            return string((istreambuf_iterator<char>(f)), istreambuf_iterator<char>());
        } catch (const exception& e) {
            return "Error reading file: " + string(e.what());
        }
    }
    string undo_last_change(const string& file_path) {
        fs::path path(file_path);
        regex pattern(path.filename().string() + R"(_v(\d+)\.bak)");
        vector<pair<int, fs::path>> versions;
        for (const auto& entry : fs::directory_iterator(backup_dir)) {
            smatch m;
            string name = entry.path().filename().string();
            if (regex_match(name, m, pattern))
                versions.emplace_back(stoi(m[1]), entry.path());
        }
        if (versions.empty()) return "No backup found.";
        auto latest = *max_element(versions.begin(), versions.end());
        fs::copy_file(latest.second, path, fs::copy_options::overwrite_existing);
        fs::remove(latest.second);
        return "Reverted " + file_path + " to version " + to_string(latest.first) + ".";
    }
};

class ToolExecutor {
public:
    AgentConfig config;
    FileManager file_manager;
    CodeExecutor executor;
    ToolExecutor(const AgentConfig& c) : config(c), file_manager(c.backup_dir), executor(c) {}
    string write_file(const string& fp, const string& content) {
        return file_manager.write_file(fp, content);
    }
    string read_file(const string& fp) { return file_manager.read_file(fp); }
    string execute_file(const string& fp, const optional<string>& stdin_data=nullopt) {
        return executor.execute_file(fp, stdin_data);
    }
    string make_directory(const string& dp) {
        fs::create_directories(dp);
        return "Directory created: " + dp;
    }
    string install_package(const string& pkg) {
        return executor.run_shell_command("pip install " + pkg, config.max_install_timeout);
    }
    string compile_cpp(const string& fp) { return executor.compile_cpp(fp); }
    string run_shell_command(const string& cmd) { return executor.run_shell_command(cmd); }
    string escape_triple_quotes(const string& text) {
        string r;
        for (size_t i = 0; i < text.size(); ++i) {
            if (i + 2 < text.size() && text[i] == '"' && text[i+1] == '"' && text[i+2] == '"') {
                size_t bs = 0, j = i;
                while (j > 0 && text[j-1] == '\\') { bs++; j--; }
                if (bs % 2 == 0) { r += "\\\"\\\"\\\""; i += 2; continue; }
            }
            r.push_back(text[i]);
        }
        return r;
    }
    string sanitize_newlines(const string& text) {
        string r;
        r.reserve(text.size());
        bool in_str = false, esc = false;
        for (char ch : text) {
            if (esc) { r.push_back(ch); esc = false; continue; }
            if (ch == '\\') { r.push_back(ch); esc = true; continue; }
            if (ch == '"' && !in_str) { in_str = true; r.push_back(ch); continue; }
            if (ch == '"' && in_str) { in_str = false; r.push_back(ch); continue; }
            if (in_str && (ch == '\n' || ch == '\r')) { r += "\\n"; continue; }
            r.push_back(ch);
        }
        return r;
    }
    static pair<bool, size_t> findClosingDelimiter(const string& text, size_t start, char quote) {
        size_t i = start;
        while (i < text.size()) {
            if (i + 2 < text.size() && text[i] == quote && text[i+1] == quote && text[i+2] == quote) {
                int bs = 0;
                size_t j = i;
                while (j > start && text[j-1] == '\\') { bs++; j--; }
                if (bs % 2 == 0) {
                    return {true, i + 3};
                }
            }
            if (text[i] == '\\' && i + 1 < text.size()) {
                i += 2;
            } else {
                ++i;
            }
        }
        return {false, text.size()};
    }
    string sanitize_triple_quotes_for_json(const string& text) {
        string result;
        result.reserve(text.size());
        for (size_t i = 0; i < text.size(); ) {
            bool triple_found = false;
            auto try_triple = [&](char quote) -> bool {
                if (i + 2 >= text.size()) return false;
                if (text[i] != quote || text[i+1] != quote || text[i+2] != quote) return false;
                size_t content_start = i + 3;
                auto [found, next_pos] = findClosingDelimiter(text, content_start, quote);
                if (!found) return false;
                string content = text.substr(content_start, next_pos - content_start - 3);
                string escaped;
                for (char c : content) {
                    switch (c) {
                        case '"':  escaped += "\\\""; break;
                        case '\n': escaped += "\\n";  break;
                        case '\\': escaped += "\\\\"; break;
                        default:   escaped += c;
                    }
                }
                result += "\"";
                result += escaped;
                result += "\"";
                i = next_pos;
                return true;
            };
            triple_found = try_triple('"') || try_triple('\'');
            if (!triple_found) {
                result += text[i];
                ++i;
            }
        }
        return result;
    }
    vector<json> extract_actions(const string& text) {
        vector<json> actions;
        string t = text;
        t = sanitize_triple_quotes_for_json(t);
        for (int pass = 0; pass < 2; ++pass) {
            if (pass == 1) {
                t = sanitize_newlines(t);
                t = escape_triple_quotes(t);
            }
            size_t i = 0;
            while (i < t.size()) {
                size_t start = t.find('{', i);
                if (start == string::npos) break;
                size_t end = start + 1;
                int depth = 1;
                bool in_str = false, esc = false;
                while (end < t.size() && depth > 0) {
                    char c = t[end];
                    if (esc) { esc = false; }
                    else if (c == '\\') { esc = true; }
                    else if (c == '"') { in_str = !in_str; }
                    else if (!in_str) {
                        if (c == '{') depth++;
                        else if (c == '}') depth--;
                    }
                    end++;
                }
                if (depth == 0) {
                    try {
                        json obj = json::parse(t.substr(start, end - start));
                        if (obj.is_object() && obj.contains("tool"))
                            actions.push_back(apply_aliases(obj));
                    } catch (...) {}
                }
                i = start + 1;
            }
            if (!actions.empty()) break;
        }
        vector<json> unique;
        set<string> seen;
        for (auto& a : actions) {
            string k = a.dump();
            if (!seen.count(k)) { seen.insert(k); unique.push_back(a); }
        }
        return unique;
    }
    json apply_aliases(json action) {
        string tool = action.value("tool", "");
        static const unordered_map<string, unordered_map<string,string>> aliases = {
            {"write_file",{{"path","file_path"},{"filename","file_path"},{"name","file_path"}}},
            {"read_file",{{"path","file_path"},{"file","file_path"}}},
            {"execute_file",{{"path","file_path"},{"file","file_path"}}},
            {"compile_cpp",{{"path","file_path"},{"file","file_path"}}},
            {"run_shell_command",{{"cmd","command"},{"shell_args","command"}}},
            {"make_directory",{{"path","dir_path"},{"dir_name","dir_path"},{"name","dir_path"},{"directory","dir_path"}}}
        };
        auto it = aliases.find(tool);
        if (it != aliases.end() && action.contains("args")) {
            auto& args = action["args"];
            for (const auto& [bad, good] : it->second) {
                if (args.contains(bad) && !args.contains(good)) {
                    args[good] = args[bad];
                    args.erase(bad);
                }
            }
        }
        return action;
    }
    optional<string> validate_content_syntax(const string& fp, const string& content) {
        string ext = fs::path(fp).extension().string();
        transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".py") {
            string r = executor.validate_syntax(content, "python");
            if (r.find("FAILED") != string::npos) return r;
        } else if (ext == ".cpp" || ext == ".cc" || ext == ".cxx") {
            string r = executor.validate_syntax(content, "cpp");
            if (r.find("FAILED") != string::npos) return r;
        }
        return nullopt;
    }
    string execute_action_dict(json action) {
        action = apply_aliases(action);
        string tool = action.value("tool", "");
        json args = action.value("args", json::object());
        static const unordered_map<string, unordered_set<string>> req = {
            {"write_file", {"file_path","content"}},
            {"read_file", {"file_path"}},
            {"execute_file", {"file_path"}},
            {"compile_cpp", {"file_path"}},
            {"run_shell_command", {"command"}},
            {"make_directory", {"dir_path"}},
            {"finish", {}}
        };
        auto rit = req.find(tool);
        if (rit != req.end()) {
            unordered_set<string> missing;
            for (const auto& k : rit->second) if (!args.contains(k)) missing.insert(k);
            if (!missing.empty()) {
                string mstr;
                for (const auto& m : missing) mstr += (mstr.empty()?"":", ") + m;
                return "Error: Missing required arguments for '" + tool + "': " + mstr;
            }
        }
        if (tool == "write_file") {
            string fp = args.value("file_path", "");
            string content = args.value("content", "");
            auto se = validate_content_syntax(fp, content);
            if (se) return "Error: " + *se + "\nFile was NOT written. Fix the syntax error and try again.";
            return write_file(fp, content);
        }
        if (tool == "read_file") return read_file(args.value("file_path", ""));
        if (tool == "execute_file") {
            optional<string> in_data;
            if (args.contains("stdin")) in_data = args["stdin"].get<string>();
            return execute_file(args.value("file_path", ""), in_data);
        }
        if (tool == "make_directory") return make_directory(args.value("dir_path", ""));
        if (tool == "install_package") return install_package(args.value("package", ""));
        if (tool == "compile_cpp") return compile_cpp(args.value("file_path", ""));
        if (tool == "run_shell_command") return run_shell_command(args.value("command", ""));
        if (tool == "finish") return "Task marked as finished. Verifying results...";
        return "Error: Tool '" + tool + "' is not recognized.";
    }
    string parse_and_run_actions(const string& text) {
        auto actions = extract_actions(text);
        if (actions.empty()) return "Error: No valid JSON tool command found in your output.";
        string result;
        for (size_t i = 0; i < actions.size(); ++i) {
            if (i > 0) result += "\n---\n";
            result += execute_action_dict(actions[i]);
        }
        return result;
    }
};

class TaskDecomposer {
    LLMBackendManager& llm;
public:
    TaskDecomposer(LLMBackendManager& l) : llm(l) {}
    vector<string> decompose(const string& task) {
        json schema = json::parse(R"({"checklist":["sub-goal 1"]})");
        string prompt = "Break this task into 3-5 sequential sub-goals. Output ONLY JSON.\nTask: " + task;
        auto [res, ok, _] = llm.structured_chat({{"user", prompt}}, schema, 0.0, 256);
        if (ok && res.is_object() && res.contains("checklist") && res["checklist"].is_array()) {
            vector<string> out;
            for (const auto& x : res["checklist"]) out.push_back(x.get<string>());
            return out;
        }
        return {task};
    }
};

class DependencyPlanner {
    LLMBackendManager& llm;
    json schema;
    map<string,string> tool_docs;
public:
    DependencyPlanner(LLMBackendManager& l) : llm(l) {
        schema = json::parse(R"xx({"actions":[{"tool":"write_file (or make_directory, etc.)","args":{"any key":"value"}}]})xx");
        tool_docs = {
            {"write_file","args: file_path (str), content (str)"},
            {"read_file","args: file_path (str)"},
            {"execute_file","args: file_path (str), stdin (str, optional)"},
            {"compile_cpp","args: file_path (str)"},
            {"run_shell_command","args: command (str)"},
            {"make_directory","args: dir_path (str)"},
            {"finish","args: (none)"}
        };
    }
    vector<json> generate_plan(const string& task, const vector<string>& expected_outputs) {
        if (expected_outputs.empty()) return {};
        string tools_desc;
        for (const auto& [t, d] : tool_docs) tools_desc += "- " + t + ": " + d + "\n";
        string prompt = "Task: " + task + "\nRequired output files: ";
        for (size_t i = 0; i < expected_outputs.size(); ++i) {
            if (i > 0) prompt += ", ";
            prompt += expected_outputs[i];
        }
        prompt += "\n\nAvailable tools and their arguments:\n" + tools_desc + "\n"
                  "Create a JSON plan with a list of actions that will accomplish the task. "
                  "Always create directories before writing files, and execute scripts that produce output files.\n"
                  "Output ONLY the JSON array of actions.";
        auto [plan, ok, _] = llm.structured_chat({{"user", prompt}}, schema, 0.0, 1024);
        if (ok && plan.is_object() && plan.contains("actions") && plan["actions"].is_array())
            return fix_files(plan["actions"], expected_outputs);
        return fallback_plan(expected_outputs);
    }
    vector<json> fallback_plan(const vector<string>& files) {
        vector<json> actions;
        set<string> seen;
        for (const auto& fp : files) {
            string parent = fs::path(fp).parent_path().string();
            if (!parent.empty() && parent != "." && !seen.count(parent)) {
                actions.push_back(json{{"tool","make_directory"},{"args",{{"dir_path",parent}}}});
                seen.insert(parent);
            }
        }
        return actions;
    }
    vector<json> fix_files(vector<json> actions, const vector<string>& expected) {
        set<string> names;
        for (const auto& p : expected) names.insert(fs::path(p).filename().string());
        for (auto& act : actions) {
            if (act.value("tool","") == "write_file") {
                string fp = act["args"].value("file_path","");
                string bn = fs::path(fp).filename().string();
                if (names.count(bn)) {
                    for (const auto& real : expected) {
                        if (fs::path(real).filename().string() == bn) {
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
    AgentConfig config;
    LLMBackendManager llm;
    ToolExecutor tool_executor;
    SemanticMemoryManager semantic_memory;
    ReflectionEngine reflection;
    TaskDecomposer decomposer;
    DependencyPlanner planner;
    vector<string> session_context;
    string last_validation_feedback;
    string last_raw_error;
    vector<string> written_files;
    vector<string> expected_outputs;
    json validation_checks;
    string current_task;
    atomic<bool> abort_flag{false};
    vector<json> successful_plan_actions;
    CodingAgent(const AgentConfig& c) : config(c), llm(c), tool_executor(c),
        semantic_memory(c), decomposer(llm), planner(llm) {}
    string system_prompt() {
        return R"EOF(You are a Senior AI Coding Agent specializing in algorithmic optimization.
Rules:
1. Communicate exclusively through JSON tool calls.
2. Format: {"tool": "name", "args": {"arg": "val"}}
3. You MAY output multiple JSON actions in a single response, separated by newlines. Each will be executed in order.
4. Available tools (use exactly these argument names):
   - write_file: file_path (str), content (str)
   - read_file: file_path (str)
   - execute_file: file_path (str), stdin (str, optional)
   - compile_cpp: file_path (str)
   - run_shell_command: command (str)
   - make_directory: dir_path (str)
   - finish: (no args)
5. For file creation tasks: write_file → execute_file → verify → finish.
6. NEVER call finish until you have EXECUTED any script that creates output files.
7. If validation fails, read the feedback and fix the exact issue mentioned.
8. Always implement complete, functional code without placeholders or comments like '# TODO'.
9. Pay close attention to VALIDATION REQUIREMENTS — your output must contain the exact verification strings requested (matching is case-insensitive).
10. CRITICAL: Python strings that span multiple lines MUST use triple quotes ("""...""") or explicit \n escapes. Single-quoted strings cannot contain literal newlines.
11. CRITICAL: Inside JSON string values, escape all double quotes with backslash (\"). Never use bare unescaped triple quotes inside a JSON value.
12. CRITICAL: Write files to the EXACT paths specified in EXPECTED OUTPUT FILES. Do NOT create extra subdirectories unless explicitly required.
13. CRITICAL: For C++ JSON output, use ONLY <nlohmann/json.hpp>. Do NOT use <json/json.h>.
14. CRITICAL: When validation checks require specific text in your script's output (e.g., 'Query successful', 'InsufficientFundsError', 'coordinates'), your code MUST explicitly print that EXACT text. Do not paraphrase or use alternative wording.
15. CRITICAL: If a validation check provides stdin input, ensure your script reads from stdin (e.g., using input() or sys.stdin) and produces the required output strings.
16. CRITICAL: Do NOT create unnecessary subdirectories like 'project/' or 'src/' unless the task explicitly requires them. Write all files to the root directory unless specified otherwise.
17. **MISSING FILE HANDLING:** If you attempt to read or execute a file and receive an error stating the file does not exist (e.g., 'cannot read source', 'No such file'), you MUST first create that file using write_file with appropriate content before you can execute it again.
18. **C++ EXECUTION WORKFLOW:** After compiling a C++ source file with compile_cpp, to run the program always use execute_file on the original .cpp (or .cc / .cxx) source file. The system will automatically recompile (if needed) and run the latest compiled executable. Do NOT attempt to run the compiled .out file directly.
)EOF";
    }
    string get_error_hint(const string& error_output) {
        string lo = error_output;
        transform(lo.begin(), lo.end(), lo.begin(), ::tolower);
        if (lo.find("syntaxerror") != string::npos && lo.find("unterminated string") != string::npos)
            return "HINT: Unterminated string literal. For multi-line strings use triple quotes (\"\"\"...\"\"\") or \\n. Single-quoted f-strings cannot span lines.";
        if (lo.find("indentationerror") != string::npos)
            return "HINT: Python indentation error. Use exactly 4 spaces per level.";
        if (lo.find("modulenotfounderror") != string::npos || lo.find("no module named") != string::npos)
            return "HINT: Missing Python module. Use the standard library or install it first.";
        if (lo.find("no such file") != string::npos)
            return "HINT: File not found. Ensure write_file succeeds before execute_file.";
        if (lo.find("compilation_error") != string::npos || lo.find("compilation_failed") != string::npos) {
            if (error_output.find("json/json.h") != string::npos)
                return "HINT: C++ compilation failed. Replace <json/json.h> with <nlohmann/json.hpp>.";
            if (error_output.find("cannot read source") != string::npos)
                return "HINT: The C++ source file does not exist yet. You must write_file first, then compile/execute.";
            return "HINT: C++ compilation failed. Ensure all required headers are included.";
        }
        return "";
    }
    json fix_action_paths(json action) {
        string tool = action.value("tool","");
        if (tool != "write_file" && tool != "execute_file" && tool != "compile_cpp" && tool != "read_file")
            return action;
        string fp = action["args"].value("file_path","");
        if (fp.empty() || expected_outputs.empty()) return action;
        string bn = fs::path(fp).filename().string();
        for (const auto& exp : expected_outputs) {
            if (fs::path(exp).filename().string() == bn && fp != exp) {
                cout << "[Agent] Redirecting " << tool << " path: " << fp << " -> " << exp << endl;
                action["args"]["file_path"] = exp;
                break;
            }
        }
        return action;
    }
    string build_progress_hint() {
        if (expected_outputs.empty() && validation_checks.empty()) return "";
        vector<string> lines = {"📋 PROGRESS CHECKLIST:"};
        for (const auto& fp : expected_outputs) {
            string ex = fs::exists(fp) ? "✅" : "❌";
            lines.push_back("  " + ex + " " + fp);
        }
        if (validation_checks.is_array()) {
            for (const auto& check : validation_checks) {
                if (check.value("type","") == "file_exists") {
                    string ex = fs::exists(check.value("path","")) ? "✅" : "❌";
                    lines.push_back("  " + ex + " File exists: " + check.value("path",""));
                } else if (check.value("type","") == "execution") {
                    string fp = check.value("file","");
                    string ex = fs::exists(fp) ? "✅" : "❌";
                    lines.push_back("  " + ex + " Script ready: " + fp);
                    if (check.contains("expect"))
                        lines.push_back("     🔍 REQUIRED OUTPUT SUBSTRING: '" + check["expect"].get<string>() + "' (case-insensitive)");
                    if (check.contains("input"))
                        lines.push_back("     📥 STDIN INPUT: " + check["input"].dump());
                }
            }
        }
        vector<string> missing;
        for (const auto& fp : expected_outputs) if (!fs::exists(fp)) missing.push_back(fp);
        if (!missing.empty())
            lines.push_back("\n➡️  NEXT: Create these missing files: " + missing[0]);
        else
            lines.push_back("\n✅ All expected files exist. Execute validation scripts and call finish.");
        string r;
        for (const auto& l : lines) r += l + "\n";
        return r;
    }
    string build_validation_requirements() {
        if (validation_checks.empty()) return "";
        vector<string> lines = {"\n🔒 VALIDATION REQUIREMENTS — YOUR CODE MUST SATISFY ALL OF THESE:"};
        for (const auto& check : validation_checks) {
            if (check.value("type","") == "file_exists")
                lines.push_back("  • File MUST exist: " + check.value("path",""));
            else if (check.value("type","") == "execution") {
                lines.push_back("  • Script " + check.value("file","") + " MUST run with exit code 0");
                if (check.contains("input")) lines.push_back("    Stdin provided: " + check["input"].dump());
                if (check.contains("expect")) lines.push_back("    Output MUST contain EXACTLY: '" + check["expect"].get<string>() + "'");
            }
        }
        string r;
        for (const auto& l : lines) r += l + "\n";
        return r;
    }
    string build_solution_from_files() {
        set<string> uniq(written_files.begin(), written_files.end());
        vector<string> parts;
        for (const auto& fp : uniq) {
            if (!fs::exists(fp)) continue;
            ifstream f(fp);
            string c((istreambuf_iterator<char>(f)), istreambuf_iterator<char>());
            parts.push_back("=== " + fp + " ===\n" + c);
        }
        string r;
        for (size_t i = 0; i < parts.size(); ++i) {
            if (i > 0) r += "\n\n";
            r += parts[i];
        }
        return r;
    }
    vector<string> execute_plan(const vector<json>& plan) {
        vector<string> observations;
        for (size_t i = 0; i < plan.size(); ++i) {
            json action = fix_action_paths(plan[i]);
            string obs = tool_executor.execute_action_dict(action);
            observations.push_back(obs);
            string tn = action.value("tool","unknown");
            session_context.push_back("Plan Step " + to_string(i+1) + ": " + tn + " → " + obs.substr(0,300));
            if (tn == "write_file") {
                string fp = action["args"].value("file_path","");
                if (!fp.empty() && find(written_files.begin(), written_files.end(), fp) == written_files.end())
                    written_files.push_back(fp);
            }
        }
        return observations;
    }
    void attempt_direct_execution() {
        if (validation_checks.empty()) return;
        set<string> needed;
        for (const auto& c : validation_checks) {
            if (c.value("type","") == "file_exists") needed.insert(c.value("path",""));
            else if (c.value("type","") == "execution") needed.insert(c.value("file",""));
        }
        vector<string> missing;
        for (const auto& f : needed) if (!f.empty() && !fs::exists(f)) missing.push_back(f);
        if (missing.empty()) return;
        string target;
        optional<string> target_stdin;
        for (auto it = written_files.rbegin(); it != written_files.rend(); ++it) {
            if (!fs::exists(*it)) continue;
            if (it->find(".py") != string::npos && it->size() >= 3 && it->substr(it->size()-3) == ".py") {
                target = *it;
                for (const auto& c : validation_checks)
                    if (c.value("type","") == "execution" && c.value("file","") == target)
                        if (c.contains("input")) target_stdin = c["input"].get<string>();
                break;
            } else if (it->find(".cpp") != string::npos || it->find(".cc") != string::npos || it->find(".cxx") != string::npos) {
                string compiled = tool_executor.executor.compile_cpp(*it);
                if (compiled.find("COMPILATION_FAILED") == string::npos) {
                    target = compiled;
                    break;
                }
            }
        }
        if (!target.empty()) {
            cout << "[Agent] 🔧 Auto-executing " << target << " to create missing files: " << missing[0] << endl;
            string result = tool_executor.executor.execute_file(target, target_stdin);
            session_context.push_back("[Auto-exec] Executed " + target + " (missing: " + missing[0] + ")\nResult: " + result.substr(0,400));
            cout << "[Agent] Auto-exec result: " << result.substr(0,200) << endl;
        }
    }
    optional<string> check_early_success() {
        if (!validator()) return nullopt;
        cout << "[Agent] ✅ Validation passed — terminating early." << endl;
        json sol;
        sol["plan_actions"] = successful_plan_actions;
        sol["file_contents"] = json::object();
        set<string> uniq(written_files.begin(), written_files.end());
        for (const auto& fp : uniq) {
            if (!fs::exists(fp)) continue;
            ifstream f(fp);
            string c((istreambuf_iterator<char>(f)), istreambuf_iterator<char>());
            sol["file_contents"][fp] = c;
        }
        semantic_memory.store(current_task, sol.dump(2));
        return "SUCCESS: Task completed and validated.";
    }
    bool validator(const vector<string>& expected_outputs_override = {},
                   const json& validation_checks_override = json(),
                   const string& task_description = "") {
        auto& checks = validation_checks_override.is_null() ? validation_checks : validation_checks_override;
        auto& outs = expected_outputs_override.empty() ? expected_outputs : expected_outputs_override;
        for (const auto& check : checks) {
            if (check.value("type","") == "file_exists") {
                if (!fs::exists(check.value("path",""))) return false;
            } else if (check.value("type","") == "execution") {
                string fp = check.value("file","");
                if (!fs::exists(fp)) return false;
                optional<string> in_data;
                if (check.contains("input")) in_data = check["input"].get<string>();
                string result = tool_executor.executor.execute_file(fp, in_data);
                if (result.find("Return code: 0") == string::npos) return false;
                if (check.contains("expect")) {
                    string expect = check["expect"];
                    string rl = result;
                    transform(rl.begin(), rl.end(), rl.begin(), ::tolower);
                    string el = expect;
                    transform(el.begin(), el.end(), el.begin(), ::tolower);
                    if (rl.find(el) == string::npos) return false;
                }
            }
        }
        for (const auto& f : outs) if (!fs::exists(f)) return false;
        return true;
    }
    bool is_critical_error(const string& observation) {
        return observation.find("Error: Syntax validation FAILED") != string::npos
            || observation.find("COMPILATION_FAILED") != string::npos
            || observation.find("Error: Tool") != string::npos
            || observation.find("Exception executing tool") != string::npos;
    }
    vector<json> extract_replayable_plan(const string& cached_solution) {
        try {
            auto parsed = json::parse(cached_solution);
            if (parsed.is_object() && parsed.contains("plan_actions")) {
                auto plan = parsed["plan_actions"];
                if (plan.is_array()) {
                    bool ok = true;
                    for (const auto& a : plan) if (!a.contains("tool")) { ok = false; break; }
                    if (ok) return plan;
                }
            }
            if (parsed.is_array()) {
                bool ok = true;
                for (const auto& a : parsed) if (!a.contains("tool")) { ok = false; break; }
                if (ok) return parsed;
            }
        } catch (...) {}
        return {};
    }
    string run(const string& task, int max_turns=6,
               const vector<string>& exp_outs = {},
               const json& val_checks = json::array()) {
        current_task = task;
        expected_outputs = exp_outs;
        validation_checks = val_checks.is_null() ? json::array() : val_checks;
        session_context.clear();
        last_validation_feedback.clear();
        last_raw_error.clear();
        written_files.clear();
        successful_plan_actions.clear();
        abort_flag = false;
        string task_hash = md5_hash(task);
        auto cached = semantic_memory.retrieve_similar(task);
        if (cached) {
            session_context.push_back("[Memory] Similar task found. Suggested solution snippet:\n" + cached->substr(0,500) + "...");
            auto plan = extract_replayable_plan(*cached);
            if (!plan.empty()) {
                cout << "[Agent] Attempting cached plan execution..." << endl;
                successful_plan_actions = plan;
                execute_plan(plan);
                auto early = check_early_success();
                if (early) return *early;
            } else {
                session_context.push_back("[Memory] Cached solution is not a replayable plan.");
            }
        }
        cout << "[Agent] Generating plan..." << endl;
        auto plan = planner.generate_plan(task, expected_outputs);
        if (!plan.empty()) {
            successful_plan_actions = plan;
            execute_plan(plan);
            cout << "[Agent] Plan executed (" << plan.size() << " steps)." << endl;
            auto early = check_early_success();
            if (early) return *early;
        }
        vector<string> hints;
        if (!expected_outputs.empty()) {
            hints.push_back("EXPECTED OUTPUT FILES: " + expected_outputs[0]);
            for (size_t i = 1; i < expected_outputs.size(); ++i) hints.back() += ", " + expected_outputs[i];
        }
        if (!validation_checks.empty()) {
            hints.push_back("VALIDATION REQUIREMENTS:");
            for (const auto& c : validation_checks) {
                if (c.value("type","") == "file_exists")
                    hints.push_back("  - File must exist: " + c.value("path",""));
                else if (c.value("type","") == "execution") {
                    string exp = "  - Script " + c.value("file","") + " must run successfully (exit code 0)";
                    if (c.contains("expect")) exp += " and output must contain: '" + c["expect"].get<string>() + "'";
                    hints.push_back(exp);
                }
            }
        }
        string file_exists_hint = "\n" + string(60,'=') + "\n"
            "⚠️  CRITICAL WORKFLOW: write_file → execute_file → verify → finish\n"
            "⚠️  DO NOT call finish until all files exist and scripts run successfully!\n";
        for (const auto& h : hints) file_exists_hint += h + "\n";
        file_exists_hint += string(60,'=') + "\n";
        for (int turn = 0; turn < max_turns; ++turn) {
            if (abort_flag) {
                cout << "[Agent] ⚠️ Abort signal received, stopping early." << endl;
                return "FAILED: Aborted due to timeout or external signal.";
            }
            if (!last_raw_error.empty()) {
                string hint = semantic_memory.get_hint_for_error(task_hash, last_raw_error);
                if (!hint.empty()) session_context.push_back("[FailureHint] " + hint);
                string local_hint = get_error_hint(last_raw_error);
                if (!local_hint.empty()) session_context.push_back(local_hint);
            }
            string progress = build_progress_hint();
            string val_reqs = build_validation_requirements();
            string history;
            int start = max(0, (int)session_context.size() - 8);
            for (int i = start; i < (int)session_context.size(); ++i)
                history += session_context[i] + "\n";
            string warning;
            if (!last_validation_feedback.empty())
                warning = "\n🚨 VALIDATION FAILED 🚨\n" + last_validation_feedback + "\n";
            string user_msg = "TASK: " + task + "\n"
                + file_exists_hint + val_reqs + "\n" + progress + "\n" + warning
                + "HISTORY:\n" + history
                + "──────────────────────────────────────\n"
                + "Turn " + to_string(turn+1) + "/" + to_string(max_turns) + " - What is your NEXT action?";
            cout << "\n[Turn " << turn+1 << "] Querying LLM..." << endl;
            auto [content, ok, err] = llm.chat({{"system", system_prompt()}, {"user", user_msg}}, 0.0, 4096);
            if (!ok) {
                cout << "[Turn " << turn+1 << "] ❌ LLM Error: " << err << endl;
                session_context.push_back("Turn " + to_string(turn+1) + " LLM Error: " + err);
                continue;
            }
            cout << "[Turn " << turn+1 << "] LLM response (first 300 chars): " << content.substr(0,300) << endl;
            auto actions = tool_executor.extract_actions(content);
            if (!actions.empty()) {
                vector<json> writes, rest;
                for (auto& act : actions) {
                    string tool = act.value("tool", "");
                    if (tool == "write_file" || tool == "make_directory")
                        writes.push_back(move(act));
                    else
                        rest.push_back(move(act));
                }
                actions = move(writes);
                actions.insert(actions.end(),
                               make_move_iterator(rest.begin()),
                               make_move_iterator(rest.end()));
            }
            if (actions.empty()) {
                cout << "[Turn " << turn+1 << "] ⚠️  No valid action extracted" << endl;
                session_context.push_back("Turn " + to_string(turn+1) + ": No valid action found in LLM output.");
                continue;
            }
            bool halted = false;
            for (size_t action_idx = 0; action_idx < actions.size(); ++action_idx) {
                if (abort_flag) return "FAILED: Aborted due to timeout or external signal.";
                string tool_name = actions[action_idx].value("tool","unknown");
                cout << "[Turn " << turn+1 << "] 🔧 Action " << action_idx+1 << "/" << actions.size() << ": " << tool_name << endl;
                json action = fix_action_paths(actions[action_idx]);
                string observation = tool_executor.execute_action_dict(action);
                cout << "[Turn " << turn+1 << "] 📤 Observation (first 200 chars): " << observation.substr(0,200) << endl;
                session_context.push_back("Turn " + to_string(turn+1) + "." + to_string(action_idx+1) + ": " + tool_name + "\n→ " + observation.substr(0,500));
                successful_plan_actions.push_back(action);
                if (is_critical_error(observation)) {
                    last_raw_error = observation;
                    semantic_memory.store_failure(task_hash, error_type_from_output(observation), observation, json::array(), get_error_hint(observation));
                    cout << "[Turn " << turn+1 << "] 🛑 Critical error, halting action batch." << endl;
                    halted = true;
                    break;
                }
                if (observation.find("Return code:") != string::npos && observation.find("Return code: 0") == string::npos) {
                    last_raw_error = observation;
                    semantic_memory.store_failure(task_hash, error_type_from_output(observation), observation, json::array(), get_error_hint(observation));
                }
                if (tool_name == "write_file") {
                    string fp = action["args"].value("file_path","");
                    if (!fp.empty() && find(written_files.begin(), written_files.end(), fp) == written_files.end())
                        written_files.push_back(fp);
                    if (!fp.empty()) cout << "[Turn " << turn+1 << "] 📝 Wrote: " << fp << endl;
                    for (const auto& check : validation_checks) {
                        if (check.value("type","") == "execution" && check.value("file","") == fp) {
                            cout << "[Turn " << turn+1 << "] 🚀 Auto-executing " << fp << " for validation..." << endl;
                            optional<string> in_data;
                            if (check.contains("input")) in_data = check["input"].get<string>();
                            string result = tool_executor.executor.execute_file(fp, in_data);
                            session_context.push_back("[Auto-exec] " + fp + " → " + result.substr(0,400));
                            cout << "[Turn " << turn+1 << "] 📤 Auto-exec result: " << result.substr(0,200) << endl;
                            if (result.find("Return code:") != string::npos && result.find("Return code: 0") == string::npos) {
                                last_raw_error = result;
                                semantic_memory.store_failure(task_hash, error_type_from_output(result), result, json::array(), get_error_hint(result));
                            }
                            if (check.contains("expect")) {
                                string expect_str = check["expect"];
                                string rl = result;
                                transform(rl.begin(), rl.end(), rl.begin(), ::tolower);
                                string el = expect_str;
                                transform(el.begin(), el.end(), el.begin(), ::tolower);
                                if (rl.find(el) == string::npos) {
                                    last_validation_feedback = "Script " + fp + " ran successfully (exit code 0) but output is MISSING the required substring: '" + expect_str + "'. Your output MUST contain this exact text. Actual output:\n" + result.substr(0,600);
                                    cout << "[Turn " << turn+1 << "] ❌ " << last_validation_feedback.substr(0,250) << endl;
                                    session_context.push_back(last_validation_feedback);
                                }
                            }
                            break;
                        }
                    }
                    auto early = check_early_success();
                    if (early) return *early;
                }
                if (tool_name == "execute_file") {
                    auto early = check_early_success();
                    if (early) return *early;
                }
                if (tool_name == "finish") {
                    cout << "[Turn " << turn+1 << "] 🏁 Finish called, attempting auto-execution fallback..." << endl;
                    attempt_direct_execution();
                    auto early = check_early_success();
                    if (early) return *early;
                    vector<string> missing_files;
                    for (const auto& f : expected_outputs) if (!fs::exists(f)) missing_files.push_back(f);
                    vector<string> missing_checks;
                    for (const auto& check : validation_checks) {
                        if (check.value("type","") == "file_exists" && !fs::exists(check.value("path","")))
                            missing_checks.push_back("❌ File not found: " + check.value("path",""));
                        else if (check.value("type","") == "execution") {
                            string fp = check.value("file","");
                            if (!fs::exists(fp)) missing_checks.push_back("❌ Script not found: " + fp);
                            else {
                                optional<string> in_data;
                                if (check.contains("input")) in_data = check["input"].get<string>();
                                string result = tool_executor.executor.execute_file(fp, in_data);
                                if (result.find("Return code: 0") == string::npos)
                                    missing_checks.push_back("❌ Script " + fp + " failed (RC ≠ 0): " + result.substr(0,150));
                                else if (check.contains("expect")) {
                                    string expect = check["expect"];
                                    string rl = result;
                                    transform(rl.begin(), rl.end(), rl.begin(), ::tolower);
                                    string el = expect;
                                    transform(el.begin(), el.end(), el.begin(), ::tolower);
                                    if (rl.find(el) == string::npos)
                                        missing_checks.push_back("❌ Script " + fp + " output missing '" + expect + "': " + result.substr(0,150));
                                }
                            }
                        }
                    }
                    string parts;
                    if (!missing_files.empty()) {
                        parts += "Missing output files: " + missing_files[0];
                        for (size_t i = 1; i < missing_files.size(); ++i) parts += ", " + missing_files[i];
                    }
                    if (!missing_checks.empty()) {
                        if (!parts.empty()) parts += "\n";
                        for (const auto& m : missing_checks) parts += m + "\n";
                    }
                    last_validation_feedback = parts.empty() ? "Unknown validation failure. Check all files exist and scripts run successfully." : parts;
                    cout << "[Turn " << turn+1 << "] ❌ VALIDATION FAILED:\n" << last_validation_feedback << endl;
                    session_context.push_back("Turn " + to_string(turn+1) + ": finish FAILED. " + last_validation_feedback);
                    halted = true;
                    break;
                }
                auto early = check_early_success();
                if (early) return *early;
            }
            if (!halted) {
                auto early = check_early_success();
                if (early) return *early;
            }
        }
        cout << "\n❌ MAX TURNS REACHED (" << max_turns << ")" << endl;
        if (!last_validation_feedback.empty() || !last_raw_error.empty()) {
            string err = last_validation_feedback.empty() ? last_raw_error : last_validation_feedback;
            semantic_memory.store_failure(task_hash, error_type_from_output(err), err, json::array(), "Task failed after max turns. Review validation requirements.");
        }
        return "FAILED: Maximum turns reached.";
    }
};

struct TestCase {
    string id;
    string name;
    string description;
    string task_prompt;
    vector<string> expected_outputs;
    json validation_checks;
    int max_turns;
    int timeout;
    TestCase(const string& i, const string& n, const string& d, const string& p,
             const vector<string>& eo, const json& vc, int mt, int to=120)
        : id(i), name(n), description(d), task_prompt(p), expected_outputs(eo),
          validation_checks(vc), max_turns(mt), timeout(to) {}
};

class TestSuite {
    vector<TestCase> cases;
public:
    TestSuite() { cases = default_cases(); }
    vector<TestCase> default_cases() {
        vector<TestCase> c;
        c.emplace_back("T1", "Automated Debugging via Log Analysis", "Automated Debugging via Log Analysis",
            R"EOF(The file memory_leak.cpp already exists and contains a segfault bug. 1. Execute the file to observe the crash. 2. Read execution_traceback.log. 3. Fix the bug so the file runs and exits with code 0. Do NOT simply replace it with unrelated code.)EOF",
            vector<string>{"memory_leak.cpp"},
            json::parse(R"([{"type":"execution","file":"memory_leak.cpp"}])"),
            6, 120);
        c.emplace_back("T2", "Concurrent Hashing", "Parallel processing with concurrent.futures",
            R"EOF(Create concurrent_hash.py scanning for .txt/.py files, calculate SHA-256 in parallel. Store {filename: hash} dict. Print execution time and verify count matches.)EOF",
            vector<string>{"concurrent_hash.py"},
            json::parse(R"([{"type":"execution","file":"concurrent_hash.py","expect":"SHA-256"}])"),
            4, 120);
        c.emplace_back("T3", "Multi-Language Data Bridge", "Multi-Language Data Bridge",
            R"EOF(Create a C++ program named generator.cpp that simulates 10^6 particle interactions and saves the state to a data.json file using the nlohmann/json library. Subsequently, create a Python script analyzer.py that reads this JSON file, calculates the mean energy of the particles, and generates a summary report. This tests the agent ability to manage dependencies across different languages and ensure data consistency between processes.)EOF",
            vector<string>{"generator.cpp", "analyzer.py"},
            json::parse(R"([{"type":"execution","file":"analyzer.py","expect":"Mean"}])"),
            6, 120);
        c.emplace_back("T4", "Concurrent Data Processing with Thread Safety", "Concurrent Data Processing with Thread Safety",
            R"EOF(Create a CSV file named "data.csv" with 10,000 rows of random integers. Implement a C++ program "data_processor.cpp" that reads this file, processes it in parallel using threads, and outputs the processed results to "output.json". Ensure thread safety with appropriate synchronization mechanisms.)EOF",
            vector<string>{"data_processor.cpp", "output.json"},
            json::parse(R"([{"type":"execution","file":"data_processor.cpp","expect":"Processed"}])"),
            8, 120);
        c.emplace_back("T5", "Secure Input Validation", "Secure Input Validation",
            R"EOF(Create a single Python script named secure_query.py that demonstrates secure SQLite querying. The script MUST: (1) Create an SQLite database file and a 'users' table with at least one row (e.g., name='John Doe'), (2) Read lines from stdin in a loop until 'exit' is received, (3) Use parameterized queries to search for users by name, (4) Print 'Query successful' when a query executes without error, (5) Attempt a common SQL injection attack on itself and show the attack is blocked. The script will be executed with stdin 'John Doe\nexit\n' and the output MUST contain 'Query successful'.)EOF",
            vector<string>{"secure_query.py"},
            json::parse(R"([{"type":"execution","file":"secure_query.py","input":"John Doe\nexit\n","expect":"Query successful"}])"),
            5, 120);
        c.emplace_back("T6", "Monte Carlo Integration", "Probabilistic numerical methods",
            R"EOF(Create a c++ program monte_carlo.cpp to estimate ∫sin(x)dx from 0 to π using 1,000,000 random points. Compare estimate to analytical value 2. Print absolute error.)EOF",
            vector<string>{"monte_carlo.cpp"},
            json::parse(R"([{"type":"execution","file":"monte_carlo.cpp","expect":"Absolute error"}])"),
            5, 120);
        c.emplace_back("T7", "Import Restructuring", "Handle directory moves and import updates",
            R"EOF(1. Create math_lib/operations.py with power(a,b) function. Create app.py importing it and printing 2**10. 2. Move operations.py to core/utils/, update imports, verify output still 1024.)EOF",
            vector<string>{"app.py", "core/utils/operations.py"},
            json::parse(R"([{"type":"execution","file":"app.py","expect":"1024"}])"),
            8, 120);
        c.emplace_back("T8", "Vending Machine OOP", "Custom exceptions and state management",
            R"EOF(Create vending.py with VendingMachine class and InsufficientFundsError exception. Demonstrate: deposit $2.00, try buying a $2.50 item, catch the InsufficientFundsError, and print the EXACT text 'InsufficientFundsError' (this exact string is required for validation). Then deposit $1 more and buy successfully. Finally print the remaining balance and inventory.)EOF",
            vector<string>{"vending.py"},
            json::parse(R"([{"type":"execution","file":"vending.py","expect":"InsufficientFundsError"}])"),
            6, 120);
        c.emplace_back("T9", "Pi Approximation", "Numerical methods and convergence",
            R"EOF(Create pi_approx.py using Leibniz formula: π = 4 * Σ((-1)^n/(2n+1)). Iterate until |approximation - math.pi| < 10^-5. Print iterations needed.)EOF",
            vector<string>{"pi_approx.py"},
            json::parse(R"([{"type":"execution","file":"pi_approx.py","expect":"iterations"}])"),
            4, 120);
        c.emplace_back("T10", "Grid BFS Pathfinding", "Graph traversal and obstacle handling",
            R"EOF(Create grid_bfs.py with a 10x10 grid containing some obstacles. Implement BFS from (0,0) to (9,9). Print the path using the EXACT text 'coordinates' (for example: 'Path coordinates: [...]') and the step count. If no path exists, print 'No path found.'.)EOF",
            vector<string>{"grid_bfs.py"},
            json::parse(R"([{"type":"execution","file":"grid_bfs.py","expect":"coordinates"}])"),
            4, 120);
        c.emplace_back("T11", "SQLite Relational Query", "Complex JOINs and subqueries",
            R"EOF(Create SQLite database with Departments and Employees tables. Insert 3 departments, 10 employees. Query: find employees earning more than their dept average. Save to high_earners.json. Use proper foreign keys.)EOF",
            vector<string>{"high_earners.json"},
            json::parse(R"([{"type":"file_exists","path":"high_earners.json"}])"),
            8, 120);
        c.emplace_back("T12", "Memoization Decorator", "Higher-order functions and performance",
            R"EOF(Create fibonacci_memoized.py with memoize decorator and apply to recursive Fibonacci. Calculate F(50) with timing. Compare to F(30) without decorator.)EOF",
            vector<string>{"fibonacci_memoized.py"},
            json::parse(R"xx([{"type":"execution","file":"fibonacci_memoized.py","expect":"F(50)"}])xx"),
            6, 120);
        c.emplace_back("T13", "Cross-File Refactoring", "Cross-File Refactoring",
            R"EOF(Task the agent with renaming a specific class or utility function in a multi-file project. For example, it must create a directory src/math, move several mathematical functions from a single utils.py into separate modules, and then update the import statements in an app.py located in the root directory. This evaluates the efficiency of the FileManager in handling path updates and recursive directory searches.)EOF",
            vector<string>{"app.py", "utils.py"},
            json::parse(R"([{"type":"execution","file":"app.py","expect":"power"}])"),
            4, 120);
        c.emplace_back("T14", "Data Cleaning Pipeline", "Handle dirty CSV data with type validation",
            R"EOF(Create raw_data.csv (Name, Age, Salary) with 6 rows including non-numeric ages/empty salaries. Create cleaner.py to filter invalid rows, calculate average salary, save to processed_data.json.)EOF",
            vector<string>{"raw_data.csv", "cleaner.py", "processed_data.json"},
            json::parse(R"([{"type":"file_exists","path":"processed_data.json"}])"),
            6, 120);
        c.emplace_back("T15", "Networked Resource Fetching", "Networked Resource Fetching",
            R"EOF(Require the agent to use the curl capability to interface with a public API to retrieve current weather or financial data. The agent must then use openssl/md5 to create a unique checksum of the response to verify data integrity before processing. This evaluates the ability to handle external libraries and asynchronous-like behavior within the execution environment.)EOF",
            vector<string>{"api_fetch.py"},
            json::parse(R"([{"type":"execution","file":"api_fetch.py","expect":"Checksum"}])"),
            4, 120);
        return c;
    }
    vector<tuple<string, bool, double, string>> run(CodingAgent& agent, const optional<vector<string>>& test_ids = nullopt) {
        vector<tuple<string, bool, double, string>> results;
        for (auto& case_ : cases) {
            if (test_ids && find(test_ids->begin(), test_ids->end(), case_.id) == test_ids->end()) continue;
            agent.abort_flag = false;
            agent.llm.reset_abort();
            agent.llm.reset_cooldowns();
            cout << "\n--- " << case_.id << " " << case_.name << " ---" << endl;
            struct CaseResult { string output; exception_ptr exc; atomic<bool> done{false}; };
            CaseResult result;
            auto start = chrono::steady_clock::now();
            thread t([&]() {
                try {
                    result.output = agent.run(case_.task_prompt, case_.max_turns, case_.expected_outputs, case_.validation_checks);
                } catch (...) {
                    result.exc = current_exception();
                }
                result.done = true;
            });
            bool timed_out = false;
            while (!result.done) {
                auto elapsed = chrono::duration_cast<chrono::seconds>(
                    chrono::steady_clock::now() - start).count();
                if (elapsed >= case_.timeout) {
                    timed_out = true;
                    agent.abort_flag = true;
                    agent.llm.abort();
                    break;
                }
                this_thread::sleep_for(chrono::milliseconds(10));
            }
            if (t.joinable()) {
                if (result.done) t.join();
                else {
                    auto join_start = chrono::steady_clock::now();
                    while (!result.done && chrono::duration_cast<chrono::seconds>(
                           chrono::steady_clock::now() - join_start).count() < 2) {
                        this_thread::sleep_for(chrono::milliseconds(10));
                    }
                    if (t.joinable()) t.detach();
                }
            }
            double dur = chrono::duration_cast<chrono::milliseconds>(
                chrono::steady_clock::now() - start).count() / 1000.0;
            if (result.exc) {
                try { rethrow_exception(result.exc); }
                catch (const exception& e) {
                    cout << "✗ Exception: " << e.what() << endl;
                    results.emplace_back(case_.id, false, dur, e.what());
                }
                continue;
            }
            if (timed_out) {
                cout << "✗ Timed out after " << case_.timeout << "s" << endl;
                results.emplace_back(case_.id, false, dur, "TIMEOUT");
                continue;
            }
            bool ok = agent.validator(case_.expected_outputs, case_.validation_checks, case_.description);
            results.emplace_back(case_.id, ok, dur, result.output.substr(max(0, (int)result.output.size()-200)));
            cout << (ok ? "✓" : "✗") << " " << dur << "s" << endl;
        }
        return results;
    }
};

string read_multiline_input(const string& prompt) {
    cout << prompt << endl;
    string line, result;
    while (getline(cin, line)) {
        if (line.empty()) break;
        if (!result.empty()) result += "\n";
        result += line;
    }
    return result;
}

void run_single_task(CodingAgent& agent) {
    string task = read_multiline_input("Enter your task (press Enter on an empty line to finish):");
    if (task.empty()) { cout << "No task entered." << endl; return; }
    int max_turns = 6;
    while (true) {
        cout << "Max turns (default 6): ";
        string in; getline(cin, in); in = trim(in);
        if (in.empty()) break;
        try { max_turns = stoi(in); break; }
        catch (...) { cout << "Invalid number. Please enter an integer." << endl; }
    }
    cout << "Expected output files (comma separated, optional): ";
    string e_in; getline(cin, e_in);
    vector<string> expected_outputs = split(e_in, ',');
    json validation_checks = json::array();
    if (!expected_outputs.empty()) {
        cout << "Add a simple execution validation? (y/n, default n): ";
        string av; getline(cin, av);
        if (trim(av) == "y") {
            for (const auto& fp : expected_outputs) {
                cout << "Expected substring in output of " << fp << " (optional): ";
                string ex; getline(cin, ex); ex = trim(ex);
                if (ex.empty()) validation_checks.push_back({{"type","execution"},{"file",fp}});
                else validation_checks.push_back({{"type","execution"},{"file",fp},{"expect",ex}});
            }
        }
    } else {
        cout << "\n⚠️  No expected output files specified. The agent will only succeed if it calls 'finish' and the last command had no errors.\n"
                "   For most tasks, you should provide at least one expected output file.\n" << endl;
    }
    cout << "\n--- Running task ---" << endl;
    string result = agent.run(task, max_turns, expected_outputs, validation_checks);
    cout << "\n=== RESULT ===\n" << result << endl;
}

void run_test_suite(CodingAgent& agent, TestSuite& suite) {
    cout << "Enter test IDs to run (comma separated, leave empty for all): ";
    string in; getline(cin, in);
    optional<vector<string>> test_ids;
    if (!trim(in).empty()) test_ids = split(in, ',');
    auto results = suite.run(agent, test_ids);
    if (!results.empty()) {
        int total = results.size(), successful = 0;
        for (const auto& [_, ok, __, ___] : results) if (ok) successful++;
        double pct = total > 0 ? (successful * 100.0 / total) : 0;
        cout << "\n📊 Summary: " << successful << "/" << total << " tests passed (" << fixed << setprecision(1) << pct << "%)" << endl;
    } else {
        cout << "No tests were run." << endl;
    }
}

int main() {
    AgentConfig config = AgentConfig::from_env();
    try {
        CodingAgent agent(config);
        TestSuite suite;
        while (true) {
            cout << "\n" << string(50,'=') << endl;
            cout << "1. Single task" << endl;
            cout << "2. Run tests" << endl;
            cout << "3. Stats" << endl;
            cout << "4. Quit" << endl;
            cout << "Choice: ";
            string choice; getline(cin, choice);
            choice = trim(choice);
            transform(choice.begin(), choice.end(), choice.begin(), ::tolower);
            if (choice == "4" || choice == "quit" || choice == "exit") {
                cout << "Goodbye." << endl;
                break;
            } else if (choice == "1") {
                run_single_task(agent);
            } else if (choice == "2") {
                run_test_suite(agent, suite);
            } else if (choice == "3") {
                agent.semantic_memory.print_stats();
            } else {
                cout << "Invalid choice. Please enter 1, 2, 3, or 4." << endl;
            }
        }
    } catch (const runtime_error& e) {
        cout << "Initialization error: " << e.what() << endl;
        cout << "Make sure at least one API key is set in environment variables:" << endl;
        cout << "  GROQ_API_KEY, OPENROUTER_API_KEY, or GOOGLE_API_KEY" << endl;
        return 1;
    }
    return 0;
}
