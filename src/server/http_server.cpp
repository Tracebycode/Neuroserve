#include "neuroserve/server/http_server.hpp"
#include "neuroserve/health/health_checker.hpp"
#include "neuroserve/metrics/metrics_registry.hpp"
#include "neuroserve/common/logger.hpp"
#include <nlohmann/json.hpp>

namespace neuroserve {

using json = nlohmann::json;

HttpServer::HttpServer(const Config& config, std::shared_ptr<InferenceService> inference_service)
    : config_(config),
      inference_service_(std::move(inference_service)),
      server_(std::make_unique<httplib::Server>()) {
    setup_routes();
}

HttpServer::~HttpServer() {
    stop();
}

void HttpServer::setup_routes() {
    // Health endpoint (Liveness)
    server_->Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.status = 200;
        res.set_content(HealthChecker::instance().liveness_json(), "application/json");
    });

    // Readiness endpoint
    server_->Get("/ready", [](const httplib::Request&, httplib::Response& res) {
        if (HealthChecker::instance().is_ready()) {
            res.status = 200;
            res.set_content(HealthChecker::instance().readiness_json(), "application/json");
        } else {
            res.status = 503;
            res.set_content(HealthChecker::instance().readiness_json(), "application/json");
        }
    });

    // Prometheus Metrics endpoint
    server_->Get("/metrics", [this](const httplib::Request&, httplib::Response& res) {
        size_t q_depth = inference_service_->queue_depth();
        size_t workers = inference_service_->active_workers();
        std::string metrics_str = MetricsRegistry::instance().to_prometheus(q_depth, workers);
        res.status = 200;
        res.set_content(metrics_str, "text/plain; version=0.0.4");
    });

    // Inference endpoint
    server_->Post("/v1/infer", [this](const httplib::Request& req, httplib::Response& res) {
        InferenceRequest inf_req;
        inf_req.request_id = ++request_counter_;
        inf_req.arrival_time = Clock::now();

        try {
            auto body_json = json::parse(req.body);
            if (!body_json.contains("model_name")) {
                inf_req.model_name = config_.model.name;
            } else {
                inf_req.model_name = body_json["model_name"].get<std::string>();
            }

            if (!body_json.contains("inputs") || !body_json["inputs"].is_array()) {
                res.status = 400;
                res.set_content(R"({"error":"Missing or invalid 'inputs' array in request body"})", "application/json");
                return;
            }

            for (const auto& item : body_json["inputs"]) {
                TensorData td;
                td.name = item.value("name", "");
                if (item.contains("shape") && item["shape"].is_array()) {
                    td.shape = item["shape"].get<std::vector<int64_t>>();
                }
                if (item.contains("data") && item["data"].is_array()) {
                    td.float_data = item["data"].get<std::vector<float>>();
                }
                inf_req.inputs.push_back(std::move(td));
            }
        } catch (const std::exception& ex) {
            res.status = 400;
            json err = {{"error", "JSON parse error: " + std::string(ex.what())}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        auto inf_res = inference_service_->infer(std::move(inf_req));

        res.status = inf_res.status_code;
        json resp_json;
        resp_json["request_id"] = inf_res.request_id;
        resp_json["success"] = inf_res.success;

        if (!inf_res.success) {
            resp_json["error"] = inf_res.error_message;
        } else {
            json outputs = json::array();
            for (const auto& out : inf_res.outputs) {
                json out_obj;
                out_obj["name"] = out.name;
                out_obj["shape"] = out.shape;
                out_obj["data"] = out.float_data;
                outputs.push_back(std::move(out_obj));
            }
            resp_json["outputs"] = std::move(outputs);
            resp_json["latency_ms"] = {
                {"queue", inf_res.queue_latency.count() / 1000.0},
                {"inference", inf_res.inference_latency.count() / 1000.0},
                {"total", inf_res.total_latency.count() / 1000.0}
            };
        }

        res.set_content(resp_json.dump(), "application/json");
    });
}

void HttpServer::start() {
    if (running_.exchange(true)) {
        return;
    }

    NS_LOG_INFO("Starting HTTP server on {}:{}", config_.server.host, config_.server.port);
    server_thread_ = std::jthread([this]() {
        server_->listen(config_.server.host.c_str(), config_.server.port);
    });
}

void HttpServer::stop() {
    if (!running_.exchange(false)) {
        return;
    }

    NS_LOG_INFO("Stopping HTTP server...");
    if (server_) {
        server_->stop();
    }
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
    NS_LOG_INFO("HTTP server stopped");
}

} // namespace neuroserve

