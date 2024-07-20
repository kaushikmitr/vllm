package main
import (
	"context"
	"fmt"
	"flag"
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
	"math/rand"


	"github.com/coocood/freecache"
	"github.com/prometheus/common/expfmt"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"github.com/prometheus/client_model/go"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	filterPb "github.com/envoyproxy/go-control-plane/envoy/extensions/filters/http/ext_proc/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	healthPb "google.golang.org/grpc/health/grpc_health_v1"
)

var (
	cacheActiveLoraModel              *freecache.Cache
	cachePendingRequestActiveAdapters *freecache.Cache
	pods                              []string
	podIPMap                          map[string]string
		//pods = []string{
		//"vllm-0.vllm-lora.default.svc.cluster.local",
		//"vllm-1.vllm-lora.default.svc.cluster.local",
		//"vllm-2.vllm-lora.default.svc.cluster.local",
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
	NumberOfPendingRequests int
}

type PendingRequestActiveAdaptersMetrics struct {
	Date                  string
	PodName               string
	PendingRequests       int
	NumberOfActiveAdapters int
}

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}
func fetchLoraMetricsFromPod(pod string, ch chan<- []ActiveLoraModelMetrics, wg *sync.WaitGroup) {
	defer wg.Done()
	ip, exists := podIPMap[pod]
	if !exists{
		log.Printf("pod %s has no corresponding ip defined", pod)
		return
	}
	url := fmt.Sprintf("http://%s/metrics", ip)
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
	var adapterList []string
	modelsDict := make(map[string]int)
	 
	for name, mf := range metricFamilies {
		if name == "vllm:active_lora_adapters" {
			for _, m := range mf.GetMetric() {
				modelName := getLabelValue(m, "active_lora_adapters")
				numberOfPendingRequests := int(m.GetGauge().GetValue())
				modelsDict[modelName] = numberOfPendingRequests
			}
		}
		if name == "vllm:info_active_adapters_info" {
			for _, metric := range mf.GetMetric() {
				for _, label := range metric.GetLabel() {
					if label.GetName() == "active_adapters" {
						if label.GetValue() != ""{
							adapterList = strings.Split(label.GetValue(), ",")
						}
					}
				}
			}
		}
	}

	for modelName, numberOfPendingRequests := range modelsDict {
		if !contains(adapterList, modelName){
			continue
		}
		loraMetric := ActiveLoraModelMetrics{
			Date:               time.Now().Format(time.RFC3339),
			PodName:            pod,
			ModelName:          modelName,
			NumberOfPendingRequests: numberOfPendingRequests,
		}
		loraMetrics = append(loraMetrics, loraMetric)
	}

	ch <- loraMetrics
}

func fetchRequestMetricsFromPod(pod string, ch chan<- []PendingRequestActiveAdaptersMetrics, wg *sync.WaitGroup) {
	defer wg.Done()

	ip, exists := podIPMap[pod]
	if !exists{
		log.Printf("pod %s has no corresponding ip defined", pod)
		return
	}
	url := fmt.Sprintf("http://%s/metrics", ip)
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
		case "vllm:info_active_adapters_info":
			for _, metric := range mf.GetMetric() {
				for _, label := range metric.GetLabel() {
					if label.GetName() == "active_adapters" {
						if label.GetValue() != ""{
							adapterCount = len(strings.Split(label.GetValue(), ","))
						}
					}
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

	// If no metrics found for the requested model, choose the pod with the least active adapters randomly
	if len(relevantMetrics) == 0 {
		minActiveAdapters := math.MaxInt
		var podsWithLeastAdapters []PendingRequestActiveAdaptersMetrics
		for _, reqMetric := range requestMetrics {
			if reqMetric.NumberOfActiveAdapters < minActiveAdapters {
				minActiveAdapters = reqMetric.NumberOfActiveAdapters
				podsWithLeastAdapters = []PendingRequestActiveAdaptersMetrics{}
			}
			if reqMetric.NumberOfActiveAdapters == minActiveAdapters {
				podsWithLeastAdapters = append(podsWithLeastAdapters, reqMetric)
			}
		}

		if len(podsWithLeastAdapters) == 0 {
			fmt.Println("Error: No pod with min adapter found")
		} else {
			targetPod = podsWithLeastAdapters[rand.Intn(len(podsWithLeastAdapters))].PodName
			fmt.Printf("Selected pod with the least active adapters: %s\n", targetPod)
		}
		return targetPod
	}

	// Find the pod with the max lora requests among the relevant metrics
	maxNumberOfPendingRequests := -1
	var bestPods []ActiveLoraModelMetrics
	for _, metric := range relevantMetrics {
			if metric.ModelName == loraAdapterRequested {
				if metric.NumberOfPendingRequests > maxNumberOfPendingRequests {
					maxNumberOfPendingRequests = metric.NumberOfPendingRequests
					bestPods = []ActiveLoraModelMetrics{}
				}
				if metric.NumberOfPendingRequests == maxNumberOfPendingRequests {
					bestPods = append(bestPods, metric)
				}
			}
	}

	if len(bestPods) > 0 {
		rand.Seed(time.Now().UnixNano())
		targetPod = bestPods[rand.Intn(len(bestPods))].PodName
		fmt.Printf("Selected pod with the highest NumberOfPendingRequests: %s\n", targetPod)
	} else {

			fmt.Printf("No pods match the requested model: %s\n")
		}

	// If the number of active Lora adapters in the selected pod is greater than the threshold, choose the pod with the least requests
	if maxNumberOfPendingRequests > threshold && bestAlternativePod != "" {
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
	threshold := 100000
	targetPod := ""

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

			log.Printf("Headers: %+v\n", h)
			log.Printf("EndOfStream: %v\n", h.RequestHeaders.EndOfStream)

			for _, n := range h.RequestHeaders.Headers.Headers {
				if strings.ToLower(n.Key) == "target-pod" {
					targetPod = n.Value
				}
			}

			if targetPod == ""{
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
		} else {
			bodyMode := filterPb.ProcessingMode_NONE

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
											Value: targetPod,
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
		}
			break

		case *extProcPb.ProcessingRequest_RequestBody:
			log.Println("--- In RequestBody processing")

			var requestBody map[string]interface{}
			if err := json.Unmarshal(v.RequestBody.Body, &requestBody); err != nil {
				log.Printf("Error unmarshaling request body: %v", err)
				break
			}

			loraAdapterRequested, ok := requestBody["model"].(string)
			if !ok {
				log.Println("model/lora-adapter not found in request body")
				break
			}

			thresholdValue, ok := requestBody["threshold"].(float64)
			if ok {
				threshold = int(thresholdValue)
			}

			if targetPod == ""{
				// Retrieve metrics from cache
				var loraMetrics []ActiveLoraModelMetrics
				var requestMetrics []PendingRequestActiveAdaptersMetrics

				for _, pod := range pods {
					loraMetric, err := getCacheActiveLoraModel(pod, loraAdapterRequested)
					if err == nil {
						loraMetrics = append(loraMetrics, *loraMetric)
					} else if err != freecache.ErrNotFound {
						log.Printf("Error fetching cacheActiveLoraModel for pod %s and lora_adapter_requested %s: %v", pod, loraAdapterRequested, err)
					}

					requestMetric, err := getCachePendingRequestActiveAdapters(pod)
					if err == nil {
						requestMetrics = append(requestMetrics, *requestMetric)
					} else if err != freecache.ErrNotFound {
						log.Printf("Error fetching cachePendingRequestActiveAdapters for pod %s: %v", pod, err)
						break
					}
				}

				fmt.Printf("Fetched loraMetrics: %+v\n", loraMetrics)
				fmt.Printf("Fetched requestMetrics: %+v\n", requestMetrics)

				targetPod = FindTargetPod(loraMetrics, requestMetrics, loraAdapterRequested, threshold)
			}
			fmt.Printf("Selected target pod: %s\n", targetPod)

			if !contains(pods, targetPod) {
				resp = &extProcPb.ProcessingResponse{
					Response: &extProcPb.ProcessingResponse_ImmediateResponse{
						ImmediateResponse: &extProcPb.ImmediateResponse{
							Status: &envoyTypePb.HttpStatus{
								Code: envoyTypePb.StatusCode_NotFound,
							},
						},
					},
				}
			}

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
			pendingQueueSize = -1
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
				for modelName, numberOfPendingRequests := range modelNames {
					metric := ActiveLoraModelMetrics{
						Date:               time.Now().Format(time.RFC3339),
						PodName:            targetPod,
						ModelName:          modelName,
						NumberOfPendingRequests: numberOfPendingRequests,
					}
					podAdapterMap[metric.PodName]++
					loraMetrics = append(loraMetrics, metric)
				}
				// Update cache with parsed values
				for _, metric := range loraMetrics {
					if err := setCacheActiveLoraModel(metric); err != nil {
						log.Printf("Error setting cache in Response Header: %v", err)
					}
				}
			}
			if pendingQueueSize >= 0{
				requestMetric := PendingRequestActiveAdaptersMetrics{
					Date:                  time.Now().Format(time.RFC3339),
					PodName:               targetPod,
					PendingRequests:       pendingQueueSize,
					NumberOfActiveAdapters: podAdapterMap[targetPod],
					}
				requestMetrics = append(requestMetrics, requestMetric)
				for _, metric := range requestMetrics {
					if err := setCachePendingRequestActiveAdapters(metric); err != nil {
						log.Printf("Error setting cache in Response Header: %v", err)
					}
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
	podsFlag := flag.String("pods", "", "Comma-separated list of pod addresses")
	podIPsFlag := flag.String("podIPs", "", "Comma-separated list of pod IPs")
	flag.Parse()

	if *podsFlag == "" || *podIPsFlag == "" {
		log.Fatal("No pods or pod IPs provided. Use the -pods and -podIPs flags to specify comma-separated lists of pod addresses and pod IPs.")
	}

	pods = strings.Split(*podsFlag, ",")
	podIPs := strings.Split(*podIPsFlag, ",")

	if len(pods) != len(podIPs) {
		log.Fatal("The number of pod addresses and pod IPs must match.")
	}

	podIPMap = make(map[string]string)
	for i := range pods {
		podIPMap[pods[i]] = podIPs[i]
	}

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