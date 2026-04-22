#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fastertransformer {

class PrefetchTraceLogger {
public:
    static PrefetchTraceLogger& instance()
    {
        static PrefetchTraceLogger instance;
        return instance;
    }

    bool enabled() const
    {
        const char* trace_path = std::getenv("PREGATED_TRACE_OUT");
        return trace_path != nullptr && trace_path[0] != '\0';
    }

    void recordLayerEvent(const std::string&                    current_layer_name,
                          const std::string&                    next_layer_name,
                          const std::vector<std::pair<int, int>>& actual_counts,
                          int                                   num_experts,
                          bool                                  is_first_moe,
                          bool                                  is_last_moe,
                          bool                                  store_prediction,
                          int64_t                               prefetch_issue_id = -1)
    {
        if (!enabled() || !startsWith(current_layer_name, "decoder::")) {
            return;
        }

        const Metadata meta = readMetadata();
        std::lock_guard<std::mutex> guard(mutex_);
        ensureOpenLocked();
        resetRequestStateLocked(meta.request_id);

        if (is_first_moe && !step_open_) {
            ++current_step_;
            step_open_ = true;
            seen_actual_layers_.clear();
            pending_predictions_.clear();
        }
        if (!step_open_) {
            ++current_step_;
            step_open_ = true;
            seen_actual_layers_.clear();
            pending_predictions_.clear();
        }

        const std::vector<int> actual_experts = flattenExperts(actual_counts);
        const std::string      actual_counts_str = joinExpertCounts(actual_counts);
        const std::string      actual_experts_str = joinInts(actual_experts);

        if (seen_actual_layers_.count(current_layer_name) == 0) {
            auto pred_it = pending_predictions_.find(current_layer_name);
            if (pred_it != pending_predictions_.end()) {
                writeConfusionLocked(meta,
                                     pred_it->second.source_layer,
                                     current_layer_name,
                                     pred_it->second.predicted_experts,
                                     actual_counts,
                                     num_experts,
                                     pred_it->second.prefetch_issue_id);
                pending_predictions_.erase(pred_it);
            }

            writeLineLocked("ACTUAL",
                            meta,
                            current_layer_name,
                            "",
                            num_experts,
                            "",
                            actual_experts_str,
                            actual_counts_str,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1.0);
            seen_actual_layers_.insert(current_layer_name);
        }

        if (store_prediction && !is_last_moe && !next_layer_name.empty()) {
            pending_predictions_[next_layer_name] =
                PendingPrediction{current_layer_name, actual_experts, prefetch_issue_id};
            writeLineLocked("PREDICTION",
                            meta,
                            current_layer_name,
                            next_layer_name,
                            num_experts,
                            actual_experts_str,
                            "",
                            "",
                            -1,
                            -1,
                            -1,
                            -1,
                            prefetch_issue_id,
                            -1,
                            -1,
                            -1,
                            -1.0);
        }

        if (is_last_moe) {
            step_open_ = false;
            seen_actual_layers_.clear();
            pending_predictions_.clear();
        }
    }

    void recordRequestSummary(double cache_hit_rate, int max_active_experts, double average_active_experts)
    {
        if (!enabled()) {
            return;
        }

        const Metadata meta = readMetadata();
        std::lock_guard<std::mutex> guard(mutex_);
        ensureOpenLocked();
        resetRequestStateLocked(meta.request_id);

        if (!stream_.is_open()) {
            return;
        }

        stream_ << "SUMMARY"
                << '\t' << sanitize(meta.trace_id)
                << '\t' << sanitize(meta.condition)
                << '\t' << sanitize(meta.request_id)
                << '\t' << sanitize(meta.domain)
                << '\t' << current_step_
                << '\t' << cache_hit_rate
                << '\t' << max_active_experts
                << '\t' << average_active_experts
                << '\t'
                << '\t'
                << '\t'
                << '\t' << -1
                << '\t' << -1
                << '\t' << -1
                << '\t' << -1
                << '\t' << -1
                << '\t' << -1
                << '\t' << -1
                << '\t' << -1
                << '\t' << -1.0
                << '\n';
        stream_.flush();
    }

    void recordPrefetchExpertEvent(const std::string& source_layer,
                                   const std::string& target_layer,
                                   int64_t            prefetch_issue_id,
                                   int                expert_id,
                                   bool               cache_hit,
                                   bool               ready_before_consume,
                                   double             stall_time_ms)
    {
        if (!enabled() || !startsWith(source_layer, "decoder::")) {
            return;
        }

        const Metadata meta = readMetadata();
        std::lock_guard<std::mutex> guard(mutex_);
        ensureOpenLocked();
        resetRequestStateLocked(meta.request_id);

        writeLineLocked("PREFETCH_EXPERT",
                        meta,
                        source_layer,
                        target_layer,
                        -1,
                        "",
                        "",
                        "",
                        -1,
                        -1,
                        -1,
                        -1,
                        prefetch_issue_id,
                        expert_id,
                        cache_hit ? 1 : 0,
                        ready_before_consume ? 1 : 0,
                        stall_time_ms);
    }

private:
    struct Metadata {
        std::string trace_id;
        std::string condition;
        std::string request_id;
        std::string domain;
    };

    struct PendingPrediction {
        std::string     source_layer;
        std::vector<int> predicted_experts;
        int64_t         prefetch_issue_id = -1;
    };

    PrefetchTraceLogger() = default;

    static bool startsWith(const std::string& value, const std::string& prefix)
    {
        return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
    }

    static std::string sanitize(std::string value)
    {
        std::replace(value.begin(), value.end(), '\t', ' ');
        std::replace(value.begin(), value.end(), '\n', ' ');
        std::replace(value.begin(), value.end(), '\r', ' ');
        return value;
    }

    static std::string readEnv(const char* key, const char* fallback)
    {
        const char* value = std::getenv(key);
        return value != nullptr && value[0] != '\0' ? value : fallback;
    }

    static Metadata readMetadata()
    {
        return Metadata{
            readEnv("PREGATED_TRACE_ID", "default-trace"),
            readEnv("PREGATED_CONDITION", "unspecified"),
            readEnv("PREGATED_REQUEST_ID", "request-unknown"),
            readEnv("PREGATED_REQUEST_DOMAIN", "domain-unknown"),
        };
    }

    static std::vector<int> flattenExperts(const std::vector<std::pair<int, int>>& expert_counts)
    {
        std::vector<int> experts;
        experts.reserve(expert_counts.size());
        for (const auto& kv : expert_counts) {
            experts.push_back(kv.first);
        }
        return experts;
    }

    static std::string joinInts(const std::vector<int>& values)
    {
        std::ostringstream oss;
        for (size_t i = 0; i < values.size(); ++i) {
            if (i != 0) {
                oss << ',';
            }
            oss << values[i];
        }
        return oss.str();
    }

    static std::string joinExpertCounts(const std::vector<std::pair<int, int>>& expert_counts)
    {
        std::ostringstream oss;
        for (size_t i = 0; i < expert_counts.size(); ++i) {
            if (i != 0) {
                oss << ',';
            }
            oss << expert_counts[i].first << ':' << expert_counts[i].second;
        }
        return oss.str();
    }

    void ensureOpenLocked()
    {
        const char* env_trace_path = std::getenv("PREGATED_TRACE_OUT");
        if (env_trace_path == nullptr || env_trace_path[0] == '\0') {
            return;
        }

        const std::string trace_path = env_trace_path;
        if (stream_.is_open() && trace_path == open_path_) {
            return;
        }

        if (stream_.is_open()) {
            stream_.close();
        }
        open_path_ = trace_path;
        header_written_ = false;
        stream_.open(open_path_, std::ios::app);

        if (stream_.is_open() && !header_written_) {
            stream_.seekp(0, std::ios::end);
            if (stream_.tellp() != std::streampos(0)) {
                header_written_ = true;
                return;
            }
            stream_ << "event_type\ttrace_id\tcondition\trequest_id\tdomain\tstep_id\tsource_layer\ttarget_layer"
                    << "\tnum_experts\tpredicted_experts\tactual_experts\tactual_counts\ttp\tfp\tfn\ttn"
                    << "\tprefetch_issue_id\texpert_id\tcache_hit\tready_before_consume\tstall_time_ms\n";
            header_written_ = true;
            stream_.flush();
        }
    }

    void resetRequestStateLocked(const std::string& request_id)
    {
        if (request_id == current_request_id_) {
            return;
        }
        current_request_id_ = request_id;
        current_step_ = -1;
        step_open_ = false;
        seen_actual_layers_.clear();
        pending_predictions_.clear();
    }

    void writeConfusionLocked(const Metadata&                     meta,
                              const std::string&                  source_layer,
                              const std::string&                  target_layer,
                              const std::vector<int>&             predicted_experts,
                              const std::vector<std::pair<int, int>>& actual_counts,
                              int                                 num_experts,
                              int64_t                             prefetch_issue_id)
    {
        std::unordered_set<int> predicted(predicted_experts.begin(), predicted_experts.end());
        std::vector<int>        actual_experts = flattenExperts(actual_counts);
        std::unordered_set<int> actual(actual_experts.begin(), actual_experts.end());

        int tp = 0;
        int fp = 0;
        int fn = 0;
        for (int expert : predicted_experts) {
            if (actual.count(expert) != 0) {
                ++tp;
            }
            else {
                ++fp;
            }
        }
        for (int expert : actual_experts) {
            if (predicted.count(expert) == 0) {
                ++fn;
            }
        }
        const int tn = std::max(0, num_experts - tp - fp - fn);

        writeLineLocked("CONFUSION",
                        meta,
                        source_layer,
                        target_layer,
                        num_experts,
                        joinInts(predicted_experts),
                        joinInts(actual_experts),
                        joinExpertCounts(actual_counts),
                        tp,
                        fp,
                        fn,
                        tn,
                        prefetch_issue_id,
                        -1,
                        -1,
                        -1,
                        -1.0);
    }

    void writeLineLocked(const char*        event_type,
                         const Metadata&    meta,
                         const std::string& source_layer,
                         const std::string& target_layer,
                         int                num_experts,
                         const std::string& predicted_experts,
                         const std::string& actual_experts,
                         const std::string& actual_counts,
                         int                tp,
                         int                fp,
                         int                fn,
                         int                tn,
                         int64_t            prefetch_issue_id,
                         int                expert_id,
                         int                cache_hit,
                         int                ready_before_consume,
                         double             stall_time_ms)
    {
        if (!stream_.is_open()) {
            return;
        }

        stream_ << event_type
                << '\t' << sanitize(meta.trace_id)
                << '\t' << sanitize(meta.condition)
                << '\t' << sanitize(meta.request_id)
                << '\t' << sanitize(meta.domain)
                << '\t' << current_step_
                << '\t' << sanitize(source_layer)
                << '\t' << sanitize(target_layer)
                << '\t' << num_experts
                << '\t' << sanitize(predicted_experts)
                << '\t' << sanitize(actual_experts)
                << '\t' << sanitize(actual_counts)
                << '\t' << tp
                << '\t' << fp
                << '\t' << fn
                << '\t' << tn
                << '\t' << prefetch_issue_id
                << '\t' << expert_id
                << '\t' << cache_hit
                << '\t' << ready_before_consume
                << '\t' << stall_time_ms
                << '\n';
        stream_.flush();
    }

    std::mutex                                      mutex_;
    std::ofstream                                   stream_;
    std::string                                     open_path_;
    bool                                            header_written_ = false;
    std::string                                     current_request_id_;
    int                                             current_step_ = -1;
    bool                                            step_open_ = false;
    std::unordered_set<std::string>                 seen_actual_layers_;
    std::unordered_map<std::string, PendingPrediction> pending_predictions_;
};

}  // namespace fastertransformer
