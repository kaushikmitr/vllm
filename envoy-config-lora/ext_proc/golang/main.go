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
	"math"

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
	cacheActiveLoraModel              *freecache.Cache
	cachePendingRequestActiveAdapters *freecache.Cache
		pods = []string{
		"vllm-0.vllm-lora.default.svc.cluster.local",
		"vllm-1.vllm-lora.default.svc.cluster.local",
		"vllm-2.vllm-lora.default.svc.cluster.local",
	}
	interval = 30 * time.Second // Update interval for fetching metrics
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
type ActiveLoraModelMetrics struct {
	Date               string
	PodName            string
	ModelName          string
	ActiveLoraAdapters int
}

type PendingRequestActiveAdaptersMetrics struct {
	Date                  string
	PodName               string
	PendingRequests       int
	NumberOfActiveAdapters int
}


func fetchLoraMetricsFromPod(pod string, ch chan<- []ActiveLoraModelMetrics, wg *sync.WaitGroup) {
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

	var loraMetrics []ActiveLoraModelMetrics
	modelsDict := make(map[string]int)

	for name, mf := range metricFamilies {
		if name == "vllm:active_lora_adapters" {
			for _, m := range mf.GetMetric() {
				modelName := getLabelValue(m, "dict_key")
				activeLoraAdapters := int(m.GetGauge().GetValue())
				modelsDict[modelName] = activeLoraAdapters
			}
		}
	}

	for modelName, activeLoraAdapters := range modelsDict {
		loraMetric := ActiveLoraModelMetrics{
			Date:               time.Now().Format(time.RFC3339),
			PodName:            pod,
			ModelName:          modelName,
			ActiveLoraAdapters: activeLoraAdapters,
		}
		loraMetrics = append(loraMetrics, loraMetric)
	}

	ch <- loraMetrics
}

func fetchRequestMetricsFromPod(pod string, ch chan<- []PendingRequestActiveAdaptersMetrics, wg *sync.WaitGroup) {
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

	var requestMetrics []PendingRequestActiveAdaptersMetrics
	pendingRequests := 0
	adapterCount := 0

	for name, mf := range metricFamilies {
		switch name {
		case "vllm:num_requests_waiting":
			for _, m := range mf.GetMetric() {
				pendingRequests += int(m.GetGauge().GetValue())
			}
		case "vllm:num_requests_running":
			for _, m := range mf.GetMetric() {
				pendingRequests += int(m.GetGauge().GetValue())
			}
		case "vllm:active_lora_adapters":
				for _, m := range mf.GetMetric() {
					modelName := getLabelValue(m, "dict_key")
					if modelName != ""{
						adapterCount++
					}
				}
		}
	}

	requestMetric := PendingRequestActiveAdaptersMetrics{
		Date:                  time.Now().Format(time.RFC3339),
		PodName:               pod,
		PendingRequests:       pendingRequests,
		NumberOfActiveAdapters: adapterCount,
	}
	requestMetrics = append(requestMetrics, requestMetric)

	ch <- requestMetrics
}

func fetchMetrics(pods []string) ([]ActiveLoraModelMetrics, []PendingRequestActiveAdaptersMetrics) {
	ch := make(chan []ActiveLoraModelMetrics)
	ch2 := make(chan []PendingRequestActiveAdaptersMetrics)
	var wg sync.WaitGroup
	var wg2 sync.WaitGroup

	for _, pod := range pods {
		wg.Add(1)
		go fetchLoraMetricsFromPod(pod, ch, &wg)
	}

	for _, pod := range pods {
		wg2.Add(1)
		go fetchRequestMetricsFromPod(pod, ch2, &wg2)
	}

	go func() {
		wg.Wait()
		close(ch)
	}()

	go func() {
		wg2.Wait()
		close(ch2)
	}()

	var allLoraMetrics []ActiveLoraModelMetrics
	var allRequestMetrics []PendingRequestActiveAdaptersMetrics
	for loraMetrics := range ch {
		allLoraMetrics = append(allLoraMetrics, loraMetrics...)
	}
	for requestMetrics := range ch2 {
		allRequestMetrics = append(allRequestMetrics, requestMetrics...)
	}
	return allLoraMetrics, allRequestMetrics
}

func getLabelValue(m *io_prometheus_client.Metric, label string) string {
	for _, l := range m.GetLabel() {
		if l.GetName() == label {
			return l.GetValue()
		}
	}
	return ""
}

func FindTargetPod(loraMetrics []ActiveLoraModelMetrics, requestMetrics []PendingRequestActiveAdaptersMetrics, loraAdapterRequested string, threshold int) string {
	var targetPod string
	bestAlternativePod := ""
	minAltRequests := math.MaxInt

	fmt.Println("Searching for the best pod...")

	// Filter metrics for the requested model
	for _, reqMetric := range requestMetrics {
		if reqMetric.PendingRequests < minAltRequests {
			minAltRequests = reqMetric.PendingRequests
			bestAlternativePod = reqMetric.PodName
		}
	}

	if loraAdapterRequested == "" {
		targetPod = bestAlternativePod
		if targetPod == "" {
			fmt.Println("Error: No pod found")
		} else {
			fmt.Printf("Selected the best alternative pod: %s with %d pending requests\n", targetPod, minAltRequests)
		}
		return targetPod
	}

	var relevantMetrics []ActiveLoraModelMetrics
	for _, metric := range loraMetrics {
		if metric.ModelName == loraAdapterRequested {
			relevantMetrics = append(relevantMetrics, metric)
		}
	}

	// If no metrics found for the requested model, choose the pod with the least active adapters
	if len(relevantMetrics) == 0 {
		minActiveAdapter := math.MaxInt
		for _, reqMetric := range requestMetrics {
			if minActiveAdapter > reqMetric.NumberOfActiveAdapters {
				minActiveAdapter = reqMetric.NumberOfActiveAdapters
				targetPod = reqMetric.PodName
			}
		}
		if targetPod == "" {
			fmt.Println("Error: No pod with min adapter found")
		} else {
			fmt.Printf("Selected pod with the least active adapters: %s\n", targetPod)
		}
		return targetPod
	}

	// Find the pod with the least active Lora adapters among the relevant metrics
	minActiveLoraAdapters := math.MaxInt
	for _, metric := range relevantMetrics {
		fmt.Printf("Checking pod: %s, model: %s, active Lora adapters: %d\n", metric.PodName, metric.ModelName, metric.ActiveLoraAdapters)
		if metric.ActiveLoraAdapters < minActiveLoraAdapters {
			minActiveLoraAdapters = metric.ActiveLoraAdapters
			targetPod = metric.PodName
			fmt.Printf("New best pod found: %s with %d active Lora adapters\n", targetPod, minActiveLoraAdapters)
		}
	}

	// If the number of active Lora adapters in the selected pod is greater than the threshold, choose the pod with the least requests
	if minActiveLoraAdapters > threshold && bestAlternativePod != "" {
		targetPod = bestAlternativePod
		fmt.Printf("Selected pod's active Lora adapters exceed threshold, selecting the best alternative pod: %s with %d pending requests\n", targetPod, minAltRequests)
	}

	if targetPod == "" {
		fmt.Println("Error: No pod found")
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


// Methods for setting and getting metrics from the cache
func setCacheActiveLoraModel(metric ActiveLoraModelMetrics) error {
	cacheKey := fmt.Sprintf("%s:%s", metric.PodName, metric.ModelName)
	cacheValue, err := json.Marshal(metric)
	if err != nil {
		return fmt.Errorf("error marshaling ActiveLoraModelMetrics for key %s: %v", cacheKey, err)
	}
	err = cacheActiveLoraModel.Set([]byte(cacheKey), cacheValue, 0)
	if err != nil {
		return fmt.Errorf("error setting cacheActiveLoraModel for key %s: %v", cacheKey, err)
	}
	fmt.Printf("Set cacheActiveLoraModel - Key: %s, Value: %s\n", cacheKey, cacheValue)
	return nil
}

func setCachePendingRequestActiveAdapters(metric PendingRequestActiveAdaptersMetrics) error {
	cacheKey := fmt.Sprintf("%s:", metric.PodName)
	cacheValue, err := json.Marshal(metric)
	if err != nil {
		return fmt.Errorf("error marshaling PendingRequestActiveAdaptersMetrics for key %s: %v", cacheKey, err)
	}
	err = cachePendingRequestActiveAdapters.Set([]byte(cacheKey), cacheValue, 0)
	if err != nil {
		return fmt.Errorf("error setting cachePendingRequestActiveAdapters for key %s: %v", cacheKey, err)
	}
	fmt.Printf("Set cachePendingRequestActiveAdapters - Key: %s, Value: %s\n", cacheKey, cacheValue)
	return nil
}

func getCacheActiveLoraModel(podName, modelName string) (*ActiveLoraModelMetrics, error) {
	cacheKey := fmt.Sprintf("%s:%s", podName, modelName)


	value, err := cacheActiveLoraModel.Get([]byte(cacheKey))



	if err != nil {
		return nil, fmt.Errorf("error fetching cacheActiveLoraModel for key %s: %v", cacheKey, err)
	}
	var metric ActiveLoraModelMetrics
	err = json.Unmarshal(value, &metric)
	if err != nil {
		return nil, fmt.Errorf("error unmarshaling ActiveLoraModelMetrics for key %s: %v", cacheKey, err)
	}
	fmt.Printf("Got cacheActiveLoraModel - Key: %s, Value: %s\n", cacheKey, value)
	return &metric, nil
}

func getCachePendingRequestActiveAdapters(podName string) (*PendingRequestActiveAdaptersMetrics, error) {
	cacheKey := fmt.Sprintf("%s:", podName)


	value, err := cachePendingRequestActiveAdapters.Get([]byte(cacheKey))



	if err != nil {
		return nil, fmt.Errorf("error fetching cachePendingRequestActiveAdapters for key %s: %v", cacheKey, err)
	}
	var metric PendingRequestActiveAdaptersMetrics
	err = json.Unmarshal(value, &metric)
	if err != nil {
		return nil, fmt.Errorf("error unmarshaling PendingRequestActiveAdaptersMetrics for key %s: %v", cacheKey, err)
	}
	fmt.Printf("Got cachePendingRequestActiveAdapters - Key: %s, Value: %s\n", cacheKey, value)
	return &metric, nil
}
// Inside the fetchMetricsPeriodically function
func fetchMetricsPeriodically(interval time.Duration) {
	for {
		loraMetrics, requestMetrics := fetchMetrics(pods)
		fmt.Printf("fetchMetricsPeriodically requestMetrics: %+v\n", requestMetrics)
		fmt.Printf("fetchMetricsPeriodically loraMetrics: %+v\n", loraMetrics)
		cacheActiveLoraModel.Clear()
		cachePendingRequestActiveAdapters.Clear()
		for _, metric := range loraMetrics {
			if err := setCacheActiveLoraModel(metric); err != nil {
				log.Printf("Error setting cache: %v", err)
			}
		}
		for _, metric := range requestMetrics {
			if err := setCachePendingRequestActiveAdapters(metric); err != nil {
				log.Printf("Error setting cache: %v", err)
			}
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
			// Retrieve metrics from cache
			var loraMetrics []ActiveLoraModelMetrics
			var requestMetrics []PendingRequestActiveAdaptersMetrics

			for _, pod := range pods {
				loraMetric, err := getCacheActiveLoraModel(pod, lora_adapter_requested)
				if err == nil {
					loraMetrics = append(loraMetrics, *loraMetric)
				} else if err != freecache.ErrNotFound {
					log.Printf("Error fetching cacheActiveLoraModel for pod %s and lora_adapter_requested %s: %v", pod, lora_adapter_requested, err)
				}

				requestMetric, err := getCachePendingRequestActiveAdapters(pod)
				if err == nil {
					requestMetrics = append(requestMetrics, *requestMetric)
				} else if err != freecache.ErrNotFound {
					log.Printf("Error fetching cachePendingRequestActiveAdapters for pod %s: %v", pod, err)
				}
			}

			fmt.Printf("Fetched loraMetrics: %+v\n", loraMetrics)
			fmt.Printf("Fetched requestMetrics: %+v\n", requestMetrics)
			targetPod = FindTargetPod(loraMetrics, requestMetrics, lora_adapter_requested, threshold)
			fmt.Printf("Selected target pod: %s\n", targetPod)

			// Increment PendingRequests and update ActiveLoraAdapters if needed
			if targetPod != "" {
				requestMetric, err := getCachePendingRequestActiveAdapters(targetPod)
				if err == nil {
					requestMetric.PendingRequests++
					if lora_adapter_requested != "" {
						requestMetric.NumberOfActiveAdapters++
					}
					if err := setCachePendingRequestActiveAdapters(*requestMetric); err != nil {
						log.Printf("Error updating PendingRequestActiveAdapters cache for pod %s: %v", targetPod, err)
					}
				} else if err != freecache.ErrNotFound {
					log.Printf("Error fetching cachePendingRequestActiveAdapters for pod %s: %v", targetPod, err)
				}

				if lora_adapter_requested != "" {
					loraMetric, err := getCacheActiveLoraModel(targetPod, lora_adapter_requested)
					if err == nil {
						loraMetric.ActiveLoraAdapters++
						if err := setCacheActiveLoraModel(*loraMetric); err != nil {
							log.Printf("Error updating ActiveLoraModelMetrics cache for pod %s and model %s: %v", targetPod, lora_adapter_requested, err)
						}
					} else if err == freecache.ErrNotFound {
						// Create new metric if not found in cache
						loraMetric = &ActiveLoraModelMetrics{
							Date:               time.Now().Format(time.RFC3339),
							PodName:            targetPod,
							ModelName:          lora_adapter_requested,
							ActiveLoraAdapters: 1,
						}
						if err := setCacheActiveLoraModel(*loraMetric); err != nil {
							log.Printf("Error creating new ActiveLoraModelMetrics cache for pod %s and model %s: %v", targetPod, lora_adapter_requested, err)
						}
					} else {
						log.Printf("Error fetching cacheActiveLoraModel for pod %s and model %s: %v", targetPod, lora_adapter_requested, err)
					}
				}
			}
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
							ClearRouteCache: true,
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
			var loraMetrics []ActiveLoraModelMetrics
			var requestMetrics []PendingRequestActiveAdaptersMetrics
			var modelNames map[string]int
			var pendingQueueSize int
			podAdapterMap := make(map[string]int)
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
					metric := ActiveLoraModelMetrics{
						Date:               time.Now().Format(time.RFC3339),
						PodName:            targetPod,
						ModelName:          modelName,
						ActiveLoraAdapters: activeLoraAdapters,
					}
					podAdapterMap[metric.PodName]++
					loraMetrics = append(loraMetrics, metric)
				}
			}
			requestMetric := PendingRequestActiveAdaptersMetrics{
				Date:                  time.Now().Format(time.RFC3339),
				PodName:               targetPod,
				PendingRequests:       pendingQueueSize,
				NumberOfActiveAdapters: podAdapterMap[targetPod],
			}
			requestMetrics = append(requestMetrics, requestMetric)

			// Update cache with parsed values
			for _, metric := range loraMetrics {
				if err := setCacheActiveLoraModel(metric); err != nil {
					log.Printf("Error setting cache in Response Header: %v", err)
				}
			}
			for _, metric := range requestMetrics {
				if err := setCachePendingRequestActiveAdapters(metric); err != nil {
					log.Printf("Error setting cache in Response Header: %v", err)
				}
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
	cacheActiveLoraModel = freecache.NewCache(1024)
	cachePendingRequestActiveAdapters = freecache.NewCache(1024)
	debug.SetGCPercent(20)

	// Start the periodic metrics fetching in a separate goroutine
	
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
