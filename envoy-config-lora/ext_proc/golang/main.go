package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"runtime/debug"
	"strings"
	"sync"
	"syscall"
	"time"
	"strconv"
	"encoding/json"

	"github.com/coocood/freecache"
	"github.com/prometheus/common/expfmt"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"github.com/prometheus/client_model/go"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	filterPb "github.com/envoyproxy/go-control-plane/envoy/extensions/filters/http/ext_proc/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	healthPb "google.golang.org/grpc/health/grpc_health_v1"
)

var (
	cache *freecache.Cache
		pods = []string{
		//"vllm-0.vllm-lora.default.svc.cluster.local",
		//"vllm-1.vllm-lora.default.svc.cluster.local",
		//"vllm-2.vllm-lora.default.svc.cluster.local",
	}
)

type server struct{}
type healthServer struct{}

func (s *healthServer) Check(ctx context.Context, in *healthPb.HealthCheckRequest) (*healthPb.HealthCheckResponse, error) {
	log.Printf("Handling grpc Check request + %s", in.String())
	return &healthPb.HealthCheckResponse{Status: healthPb.HealthCheckResponse_SERVING}, nil
}

func (s *healthServer) Watch(in *healthPb.HealthCheckRequest, srv healthPb.Health_WatchServer) error {
	return status.Error(codes.Unimplemented, "Watch is not implemented")
}
type ModelMetrics struct {
	Date               string
	PodName            string
	ModelName          string
	ActiveLoraAdapters int
	PendingRequests    int
}

func fetchMetricsFromPod(pod string, ch chan<- []ModelMetrics, wg *sync.WaitGroup) {
	defer wg.Done()

	url := fmt.Sprintf("http://%s:8000/metrics", pod)
	resp, err := http.Get(url)
	if err != nil {
		log.Printf("failed to fetch metrics from %s: %v", pod, err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		log.Printf("unexpected status code from %s: %v", pod, resp.StatusCode)
		return
	}

	parser := expfmt.TextParser{}
	metricFamilies, err := parser.TextToMetricFamilies(resp.Body)
	if err != nil {
		log.Printf("failed to parse metrics from %s: %v", pod, err)
		return
	}

	

	var metrics []ModelMetrics
	modelsDict := make(map[string]int)  // Create a dictionary for model names and activeLoraAdapters
	pendingRequests := 0


	for name, mf := range metricFamilies {
		var activeLoraAdapters int
		var modelName string
		switch name {
		case "vllm:active_lora_adapters":
			for _, m := range mf.GetMetric() {
				modelName = getLabelValue(m, "dict_key")
				activeLoraAdapters = int(m.GetGauge().GetValue())
				modelsDict[modelName] = activeLoraAdapters  // Update the dictionary
				//log.Printf("Pod: %s, Metric: %s, Model: %s, ActiveLoraAdapters: %d", pod, name, modelName, activeLoraAdapters)
			}
		case "vllm:num_requests_waiting":
			for _, m := range mf.GetMetric() {
				pendingRequests = int(m.GetGauge().GetValue())
				//log.Printf("Pod: %s, Metric: %s, PendingRequests: %d", pod, name, pendingRequests)
			}
		}
	}
	for modelName, activeLoraAdapters := range modelsDict {
		metric := ModelMetrics{
			Date:               time.Now().Format(time.RFC3339),
			PodName:            pod,
			ModelName:          modelName,
			ActiveLoraAdapters: activeLoraAdapters,
			PendingRequests:    pendingRequests,
		}
		metrics = append(metrics, metric)
	}
	metric := ModelMetrics{
		Date:               time.Now().Format(time.RFC3339),
		PodName:            pod,
		ModelName:          "",
		ActiveLoraAdapters: 0,
		PendingRequests:    pendingRequests,
	}
	metrics = append(metrics, metric)

	ch <- metrics
}

func fetchMetrics(pods []string) []ModelMetrics {
	ch := make(chan []ModelMetrics)
	var wg sync.WaitGroup

	for _, pod := range pods {
		wg.Add(1)
		go fetchMetricsFromPod(pod, ch, &wg)
	}

	go func() {
		wg.Wait()
		close(ch)
	}()

	var allMetrics []ModelMetrics
	for metrics := range ch {
		allMetrics = append(allMetrics, metrics...)
	}

	return allMetrics
}

func getLabelValue(m *io_prometheus_client.Metric, label string) string {
	for _, l := range m.GetLabel() {
		if l.GetName() == label {
			return l.GetValue()
		}
	}
	return ""
}

// FindTargetPod finds the best pod based on the provided metrics, requested model, and threshold
func FindTargetPod(metrics []ModelMetrics, loraAdapterRequested string, threshold int) string {
	var targetPod string
	minRequests := int(^uint(0) >> 1) // Initialize to max int value
	bestAlternativePod := ""
	minAltRequests := int(^uint(0) >> 1) // Initialize to max int value
	allZeroRequests := true

	fmt.Println("Searching for the best pod...")
	for _, metric := range metrics {
		if metric.ModelName == loraAdapterRequested && metric.ActiveLoraAdapters > 0 && metric.PendingRequests > 0 {
			allZeroRequests = false
			break
		}
	}

	for _, metric := range metrics {
		fmt.Printf("Checking pod: %s, model: %s, pending requests: %d\n", metric.PodName, metric.ModelName, metric.PendingRequests)

		if loraAdapterRequested == "" {
			// If no specific model is requested, find the pod with the least pending requests
			if metric.PendingRequests < minRequests {
				minRequests = metric.PendingRequests
				targetPod = metric.PodName
				fmt.Printf("New best pod found: %s with %d pending requests\n", targetPod, minRequests)
			}
		} else {
			// Specific model requested
			if metric.ModelName == loraAdapterRequested && metric.ActiveLoraAdapters > 0 {
				if metric.PendingRequests < minRequests {
					minRequests = metric.PendingRequests
					targetPod = metric.PodName
					fmt.Printf("New best pod found: %s with %d pending requests\n", targetPod, minRequests)
				}
			} else if allZeroRequests && metric.ModelName == loraAdapterRequested {
				if metric.PendingRequests < minRequests {
					minRequests = metric.PendingRequests
					targetPod = metric.PodName
					fmt.Printf("New best pod found: %s with %d pending requests\n", targetPod, minRequests)
				}
			} else {
				// Keep track of the pod with the least pending requests as a fallback
				if metric.PendingRequests < minAltRequests {
					minAltRequests = metric.PendingRequests
					bestAlternativePod = metric.PodName
				}
			}
		}
	}

	// If the number of requests in the selected pod is greater than the threshold, choose the pod with the least requests
	if minRequests > threshold && bestAlternativePod != "" {
		targetPod = bestAlternativePod
		fmt.Printf("Selected pod's requests exceed threshold, selecting the best alternative pod: %s with %d pending requests\n", targetPod, minAltRequests)
	}

	if targetPod == "" {
		fmt.Printf("Error: No pod found\n")
	}

	return targetPod
}

func extractPodName(dns string) string {
	parts := strings.Split(dns, ".")
	if len(parts) > 0 {
		return parts[0]
	}
	return ""
}
func fetchMetricsPeriodically(interval time.Duration) {
	for {
		metrics := fetchMetrics(pods)
		for _, metric := range metrics {
			cacheKey := fmt.Sprintf("%s:%s", metric.PodName, metric.ModelName)
			cacheValue := fmt.Sprintf("date:%s,active_adapters:%d,pending_queue:%d", metric.Date, metric.ActiveLoraAdapters, metric.PendingRequests)
			cache.Set([]byte(cacheKey), []byte(cacheValue), int(interval.Seconds()))
			log.Printf("Cache updated - Key: %s, Value: %s", cacheKey, cacheValue)
		}
		time.Sleep(interval)
	}
}

func (s *server) Process(srv extProcPb.ExternalProcessor_ProcessServer) error {

	log.Println(" ")
	log.Println(" ")
	log.Println("Started process:  -->  ")

	ctx := srv.Context()

	//contentType := ""
	lora_adapter_requested := ""
	threshold := 100000
	targetPod := "vllm-x"

	for {

		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		req, err := srv.Recv()

		if err == io.EOF {
			return nil
		}

		if err != nil {
			return status.Errorf(codes.Unknown, "cannot receive stream request: %v", err)
		}

		log.Println(" ")
		log.Println(" ")
		log.Println("Got stream:  -->  ")

		resp := &extProcPb.ProcessingResponse{}

		switch v := req.Request.(type) {

		case *extProcPb.ProcessingRequest_RequestHeaders:

			log.Println("--- In RequestHeaders processing ...")
			r := req.Request
			h := r.(*extProcPb.ProcessingRequest_RequestHeaders)

			//log.Printf("Request: %+v\n", r)
			log.Printf("Headers: %+v\n", h)
			log.Printf("EndOfStream: %v\n", h.RequestHeaders.EndOfStream)

			// List of backend pod addresses. Replace with actual pod addresses or make configurable.
			//fmt.Printf("Pods to check: %v\n", pods)
			

			for _, n := range h.RequestHeaders.Headers.Headers {
				//if strings.ToLower(n.Key) == "content-type" {
				//	contentType = n.Value
				//}
				if strings.ToLower(n.Key) == "lora-adapter" {
					lora_adapter_requested = n.Value
				}
				if strings.ToLower(n.Key) == "threshold" {
					t, err := strconv.Atoi(n.Value)
					if err != nil {
						fmt.Printf("Error converting threshold value: %n.Value\n", err)
					} else {
						threshold = t
					}

				}
			}
			//fmt.Println("Fetching metrics from pods...")
			//metrics := fetchMetrics(pods)
			// Retrieve metrics from cache
			var metrics []ModelMetrics
			for _, pod := range pods {
				cacheKey := fmt.Sprintf("%s:%s", pod, lora_adapter_requested)
				value, err := cache.Get([]byte(cacheKey))
				if err == nil {
					var metric ModelMetrics
					fmt.Scan(string(value), "date:%s,active_adapters:%d,pending_queue:%d", &metric.Date, &metric.ActiveLoraAdapters, &metric.PendingRequests)
					metric.PodName = pod 
					metric.ModelName = lora_adapter_requested
					metrics = append(metrics, metric)
				}
				if lora_adapter_requested != "" {
					cacheKey := fmt.Sprintf("%s:%s", pod, "")
					value, err := cache.Get([]byte(cacheKey))
					if err == nil {
						var metric ModelMetrics
						fmt.Scan(string(value), "date:%s,active_adapters:%d,pending_queue:%d", &metric.Date, &metric.ActiveLoraAdapters, &metric.PendingRequests)
						metric.PodName = pod 
						metric.ModelName = ""
						metrics = append(metrics, metric)
					}
					}
			}
			fmt.Printf("Fetched metrics: %+v\n", metrics)
			targetPod = FindTargetPod(metrics, lora_adapter_requested, threshold)
			fmt.Printf("Selected target pod: %s\n", targetPod)

			bodyMode := filterPb.ProcessingMode_BUFFERED

			resp = &extProcPb.ProcessingResponse{
				Response: &extProcPb.ProcessingResponse_RequestHeaders{
					RequestHeaders: &extProcPb.HeadersResponse{
						Response: &extProcPb.CommonResponse{
							HeaderMutation: &extProcPb.HeaderMutation{
								SetHeaders: []*configPb.HeaderValueOption{
									{
										Header: &configPb.HeaderValue{
											Key:   "x-went-into-req-headers",
											Value: "true",
										},
									},
									{
										Header: &configPb.HeaderValue{
											Key:   "target_pod",
											Value: extractPodName(targetPod),
										},
									},

								},
							},
						},
					},
				},
				ModeOverride: &filterPb.ProcessingMode{
					ResponseHeaderMode: filterPb.ProcessingMode_SEND,
					RequestBodyMode:    bodyMode,
				},
			}

			// Print final headers being sent
			fmt.Println("Final headers being sent:")
			for _, header := range resp.GetRequestHeaders().GetResponse().GetHeaderMutation().GetSetHeaders() {
				fmt.Printf("%s: %s\n", header.GetHeader().GetKey(), header.GetHeader().GetValue())
			}

			break

		case *extProcPb.ProcessingRequest_RequestBody:

			log.Println("--- In RequestBody processing")
			//r := req.Request
			//b := r.(*extProcPb.ProcessingRequest_RequestBody)

			//log.Printf("Request: %+v\n", r)
			//log.Printf("Body: %+v\n", b)
			//log.Printf("EndOfStream: %v\n", b.RequestBody.EndOfStream)
			//log.Printf("Content Type: %v\n", contentType)
			//log.Printf("target pod: %v\n", targetPod)

			resp = &extProcPb.ProcessingResponse{
				Response: &extProcPb.ProcessingResponse_RequestBody{
					RequestBody: &extProcPb.BodyResponse{
						Response: &extProcPb.CommonResponse{
							HeaderMutation: &extProcPb.HeaderMutation{
								SetHeaders: []*configPb.HeaderValueOption{
									{
										Header: &configPb.HeaderValue{
											Key:   "x-went-into-req-body",
											Value: "true",
										},
									},
								},
							},
						},
					},
				},
			}

			break

		case *extProcPb.ProcessingRequest_ResponseHeaders:

			log.Println("--- In ResponseHeaders processing")
			r := req.Request
			h := r.(*extProcPb.ProcessingRequest_ResponseHeaders)

			//log.Printf("Request: %+v\n", r)
			log.Printf("Headers: %+v\n", h)
			//log.Printf("Content Type: %v\n", contentType)

			// Retrieve and parse metrics from response headers
			var metrics []ModelMetrics
			var modelNames map[string]int
			var pendingQueueSize int

			for _, header := range h.ResponseHeaders.Headers.Headers {
				switch header.Key {
				case "active_lora_adapters":
					err := json.Unmarshal([]byte(header.Value), &modelNames)
					if err != nil {
						log.Printf("Error parsing model_names: %v", err)
					}
				case "pending_queue_size":
					var err error
					pendingQueueSize, err = strconv.Atoi(header.Value)
					if err != nil {
						log.Printf("Error converting pending_queue_size: %v", err)
					}
				}
			}

			if modelNames != nil {
				for modelName, activeLoraAdapters := range modelNames {
					metric := ModelMetrics{
						Date:               time.Now().Format(time.RFC3339),
						PodName:            targetPod,
						ModelName:          modelName,
						ActiveLoraAdapters: activeLoraAdapters,
						PendingRequests:    pendingQueueSize,
					}
					metrics = append(metrics, metric)
				}
			}
			metric := ModelMetrics{
				Date:               time.Now().Format(time.RFC3339),
				PodName:            targetPod,
				ModelName:          "",
				ActiveLoraAdapters: 0,
				PendingRequests:    pendingQueueSize,
			}
			metrics = append(metrics, metric)

			// Update cache with parsed values
			for _, metric := range metrics {
				cacheKey := fmt.Sprintf("%s:%s", metric.PodName, metric.ModelName)
				cacheValue := fmt.Sprintf("date:%s,active_adapters:%d,pending_queue:%d", metric.Date, metric.ActiveLoraAdapters, metric.PendingRequests)
				cache.Set([]byte(cacheKey), []byte(cacheValue), int(30*time.Second.Seconds()))
				log.Printf("Cache updated at response header - Key: %s, Value: %s", cacheKey, cacheValue)
			}

			resp = &extProcPb.ProcessingResponse{
				Response: &extProcPb.ProcessingResponse_ResponseHeaders{
					ResponseHeaders: &extProcPb.HeadersResponse{
						Response: &extProcPb.CommonResponse{
							HeaderMutation: &extProcPb.HeaderMutation{
								SetHeaders: []*configPb.HeaderValueOption{
									{
										Header: &configPb.HeaderValue{
											Key:   "x-went-into-resp-headers",
											Value: "true",
										},
									},
									{
										Header: &configPb.HeaderValue{
											Key:   "target_pod",
											Value: targetPod,
										},
									},
								},
							},
						},
					},
				},
			}

			break

		default:
			log.Printf("Unknown Request type %+v\n", v)
		}

		if err := srv.Send(resp); err != nil {
			log.Printf("send error %v", err)
		}
	}
}

func main() {

	// cache init
	cache = freecache.NewCache(1024)
	debug.SetGCPercent(20)

	// Start the periodic metrics fetching in a separate goroutine
	interval := 30 * time.Second // Update interval for fetching metrics
	go fetchMetricsPeriodically(interval)

	// grpc server init
	lis, err := net.Listen("tcp", ":50051")
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	s := grpc.NewServer()

	extProcPb.RegisterExternalProcessorServer(s, &server{})
	healthPb.RegisterHealthServer(s, &healthServer{})

	log.Println("Starting gRPC server on port :50051")

	// shutdown
	var gracefulStop = make(chan os.Signal)
	signal.Notify(gracefulStop, syscall.SIGTERM)
	signal.Notify(gracefulStop, syscall.SIGINT)
	go func() {
		sig := <-gracefulStop
		log.Printf("caught sig: %+v", sig)
		log.Println("Wait for 1 second to finish processing")
		time.Sleep(1 * time.Second)
		os.Exit(0)
	}()

	s.Serve(lis)
}

