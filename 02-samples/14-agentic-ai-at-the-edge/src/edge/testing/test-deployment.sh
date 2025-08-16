#!/bin/bash
# test-deployment.sh - Test script for edge deployment validation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_test() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

print_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

print_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
}

print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# Test 1: Check if container is running
test_container_running() {
    print_test "Checking if container is running..."
    
    if docker ps --format "table {{.Names}}" | grep -q "strands-edge-personal-assistant"; then
        print_pass "Container is running"
        return 0
    else
        print_fail "Container is not running"
        return 1
    fi
}

# Test 2: Check health status
test_health_status() {
    print_test "Checking container health status..."
    
    local health_status=$(docker inspect strands-edge-personal-assistant --format='{{.State.Health.Status}}' 2>/dev/null || echo "unknown")
    
    case "$health_status" in
        "healthy")
            print_pass "Container is healthy"
            return 0
            ;;
        "starting")
            print_test "Container is still starting up..."
            return 1
            ;;
        "unhealthy")
            print_fail "Container is unhealthy"
            return 1
            ;;
        *)
            print_fail "Unknown health status: $health_status"
            return 1
            ;;
    esac
}

# Test 3: Check if llama.cpp server is responding
test_server_response() {
    print_test "Testing llama.cpp server response..."
    
    if curl -s -f "http://localhost:8080/health" >/dev/null 2>&1; then
        print_pass "Server is responding on port 8080"
        return 0
    else
        print_fail "Server is not responding on port 8080"
        return 1
    fi
}

# Test 4: Check model files
test_model_files() {
    print_test "Checking if model files are present..."
    
    local model_path=$(docker exec strands-edge-personal-assistant printenv MODEL_PATH 2>/dev/null || echo "/app/models/qwen2.5-omni-7b.Q4_K_M.gguf")
    
    if docker exec strands-edge-personal-assistant test -f "$model_path" 2>/dev/null; then
        local model_size=$(docker exec strands-edge-personal-assistant du -h "$model_path" 2>/dev/null | cut -f1)
        print_pass "Model file exists: $(basename $model_path) ($model_size)"
        return 0
    else
        print_fail "Model file not found: $model_path"
        return 1
    fi
}

# Test 5: Check resource usage
test_resource_usage() {
    print_test "Checking resource usage..."
    
    local stats=$(docker stats --no-stream --format "{{.CPUPerc}},{{.MemUsage}}" strands-edge-personal-assistant 2>/dev/null)
    
    if [ -n "$stats" ]; then
        local cpu_percent=$(echo "$stats" | cut -d',' -f1 | sed 's/%//')
        local mem_usage=$(echo "$stats" | cut -d',' -f2)
        
        print_pass "Resource usage - CPU: ${cpu_percent}%, Memory: $mem_usage"
        
        # Check if CPU usage is reasonable (under 90%)
        if (( $(echo "$cpu_percent < 90" | bc -l) )); then
            print_pass "CPU usage is within normal range"
        else
            print_fail "CPU usage is very high: ${cpu_percent}%"
        fi
        
        return 0
    else
        print_fail "Could not retrieve resource usage"
        return 1
    fi
}

# Test 6: Simple API test
test_simple_api() {
    print_test "Testing simple inference API..."
    
    local test_payload='{
        "messages": [
            {"role": "user", "content": "Say hello"}
        ],
        "model": "default",
        "max_tokens": 10,
        "temperature": 0.7
    }'
    
    local response=$(curl -s -X POST "http://localhost:8080/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$test_payload" 2>/dev/null)
    
    if [ -n "$response" ] && echo "$response" | grep -q "choices"; then
        print_pass "API is responding correctly"
        return 0
    else
        print_fail "API test failed or returned unexpected response"
        return 1
    fi
}

# Test 7: Check logs for errors
test_logs_for_errors() {
    print_test "Checking logs for critical errors..."
    
    local error_count=$(docker logs strands-edge-personal-assistant 2>&1 | grep -i "error\|fatal\|exception" | wc -l)
    
    if [ "$error_count" -eq 0 ]; then
        print_pass "No critical errors found in logs"
        return 0
    else
        print_fail "Found $error_count potential errors in logs"
        echo "Recent errors:"
        docker logs --tail 10 strands-edge-personal-assistant 2>&1 | grep -i "error\|fatal\|exception" | head -5
        return 1
    fi
}

# Main test runner
run_all_tests() {
    print_header "EDGE DEPLOYMENT VALIDATION TESTS"
    
    local total_tests=0
    local passed_tests=0
    local failed_tests=0
    
    # List of tests to run
    tests=(
        "test_container_running"
        "test_health_status" 
        "test_server_response"
        "test_model_files"
        "test_resource_usage"
        "test_simple_api"
        "test_logs_for_errors"
    )
    
    # Run each test
    for test_func in "${tests[@]}"; do
        total_tests=$((total_tests + 1))
        
        if $test_func; then
            passed_tests=$((passed_tests + 1))
        else
            failed_tests=$((failed_tests + 1))
        fi
        
        echo ""
    done
    
    # Summary
    print_header "TEST RESULTS SUMMARY"
    echo -e "Total Tests:  $total_tests"
    echo -e "${GREEN}Passed:       $passed_tests${NC}"
    echo -e "${RED}Failed:       $failed_tests${NC}"
    
    if [ "$failed_tests" -eq 0 ]; then
        echo ""
        print_pass "🎉 All tests passed! Deployment is healthy."
        echo ""
        echo "Your Personal Assistant is ready to use:"
        echo "  ./setup.sh connect"
        return 0
    else
        echo ""
        print_fail "❌ Some tests failed. Check the issues above."
        echo ""
        echo "Troubleshooting steps:"
        echo "  1. Check logs:     docker logs strands-edge-personal-assistant"
        echo "  2. Check status:   ./setup.sh status"
        echo "  3. Restart:        ./setup.sh restart"
        echo "  4. Rebuild:        ./setup.sh rebuild"
        return 1
    fi
}

# Quick test function
quick_test() {
    print_header "QUICK DEPLOYMENT TEST"
    
    if test_container_running && test_health_status && test_server_response; then
        print_pass "✅ Quick test passed - deployment appears healthy"
        return 0
    else
        print_fail "❌ Quick test failed - check deployment"
        return 1
    fi
}

# Parse command line arguments
case "${1:-full}" in
    "full"|"all")
        run_all_tests
        ;;
    "quick"|"basic")
        quick_test
        ;;
    "health")
        test_health_status
        ;;
    "api")
        test_simple_api
        ;;
    "resources")
        test_resource_usage
        ;;
    "logs")
        test_logs_for_errors
        ;;
    "help"|"-h"|"--help")
        echo "Edge Deployment Test Script"
        echo ""
        echo "Usage: $0 [TEST_TYPE]"
        echo ""
        echo "Test Types:"
        echo "  full, all      Run all validation tests (default)"
        echo "  quick, basic   Run basic health checks only"
        echo "  health         Check container health status"
        echo "  api            Test API endpoint"
        echo "  resources      Check resource usage"
        echo "  logs           Check for errors in logs"
        echo "  help           Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0               # Run all tests"
        echo "  $0 quick         # Quick health check"
        echo "  $0 api           # Test API only"
        ;;
    *)
        echo "Unknown test type: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac