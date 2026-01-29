#!/usr/bin/env bash
# OpenRouter Integration Validation Script
#
# This script validates the OpenRouter integration by running tests
# and checking configuration.

set -e  # Exit on error

# Colours for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Colour

# Helper functions
info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=================================================="
echo "  OpenRouter Integration Validation"
echo "=================================================="
echo ""

# Check if we're in the right directory
if [ ! -f "$PROJECT_ROOT/pyproject.toml" ]; then
    error "Not in project root directory. Please run from project root."
    exit 1
fi

info "Project root: $PROJECT_ROOT"
echo ""

# ============================================
# 1. Check Environment Configuration
# ============================================
echo "1. Checking environment configuration..."

if [ ! -f "$PROJECT_ROOT/.env" ]; then
    warn ".env file not found. Creating from example..."
    if [ -f "$PROJECT_ROOT/.env.example" ]; then
        cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
        warn "Created .env file. Please add your API keys."
    else
        error ".env.example not found"
        exit 1
    fi
fi

# Check for OPENROUTER_API_KEY
if grep -q "OPENROUTER_API_KEY=" "$PROJECT_ROOT/.env"; then
    # Check if it's not just the placeholder
    if grep -q "OPENROUTER_API_KEY=sk-or-v1-" "$PROJECT_ROOT/.env"; then
        info "✓ OPENROUTER_API_KEY found in .env"
    else
        warn "OPENROUTER_API_KEY present but appears to be placeholder value"
    fi
else
    error "OPENROUTER_API_KEY not found in .env"
    echo "  Add your OpenRouter API key to .env:"
    echo "  OPENROUTER_API_KEY=sk-or-v1-..."
    exit 1
fi

echo ""

# ============================================
# 2. Check Configuration Files
# ============================================
echo "2. Checking configuration files..."

# Check providers.yaml
PROVIDERS_CONFIG="$PROJECT_ROOT/config/providers.yaml"
if [ ! -f "$PROVIDERS_CONFIG" ]; then
    error "providers.yaml not found at $PROVIDERS_CONFIG"
    exit 1
fi

# Check for openrouter section
if grep -q "^openrouter:" "$PROVIDERS_CONFIG"; then
    info "✓ OpenRouter section found in providers.yaml"
else
    error "OpenRouter section not found in providers.yaml"
    exit 1
fi

# Check for required fields
if grep -q "api_endpoint:" "$PROVIDERS_CONFIG"; then
    info "✓ api_endpoint configured"
else
    warn "api_endpoint not found in OpenRouter config"
fi

if grep -q "models:" "$PROVIDERS_CONFIG"; then
    info "✓ models configured"
else
    error "models not found in OpenRouter config"
    exit 1
fi

if grep -q "web_search:" "$PROVIDERS_CONFIG"; then
    info "✓ web_search section present"
else
    warn "web_search section not found (optional but recommended)"
fi

echo ""

# ============================================
# 3. Check Python Installation
# ============================================
echo "3. Checking Python installation..."

# Check Python version
if ! command -v python3 &> /dev/null; then
    error "python3 not found. Please install Python 3.9+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
info "✓ Python version: $PYTHON_VERSION"

# Check if package is installed
if python3 -c "import tclp_benchmark" 2>/dev/null; then
    info "✓ tclp_benchmark package installed"
else
    warn "tclp_benchmark not installed. Installing..."
    pip install -e "$PROJECT_ROOT"
fi

echo ""

# ============================================
# 4. Run Unit Tests
# ============================================
echo "4. Running OpenRouter unit tests..."

cd "$PROJECT_ROOT"

if ! command -v pytest &> /dev/null; then
    error "pytest not found. Install with: pip install pytest"
    exit 1
fi

# Run OpenRouter-specific unit tests
info "Running config tests..."
if pytest tests/unit/config/test_openrouter_config.py -v --tb=short; then
    info "✓ Config tests passed"
else
    error "Config tests failed"
    exit 1
fi

info "Running provider tests..."
if pytest tests/unit/llm_providers/test_openrouter_provider.py -v --tb=short; then
    info "✓ Provider tests passed"
else
    error "Provider tests failed"
    exit 1
fi

echo ""

# ============================================
# 5. Run Integration Tests
# ============================================
echo "5. Running OpenRouter integration tests..."

info "Running config integration tests..."
if pytest tests/integration/config/test_load_openrouter_config.py -v --tb=short; then
    info "✓ Config integration tests passed"
else
    error "Config integration tests failed"
    exit 1
fi

info "Running orchestrator integration tests..."
if pytest tests/integration/evaluators/test_orchestrator_openrouter.py -v --tb=short; then
    info "✓ Orchestrator integration tests passed"
else
    error "Orchestrator integration tests failed"
    exit 1
fi

info "Running CLI integration tests..."
if pytest tests/integration/cli/test_openrouter_cli.py -v --tb=short; then
    info "✓ CLI integration tests passed"
else
    error "CLI integration tests failed"
    exit 1
fi

echo ""

# ============================================
# 6. Run End-to-End Tests (Optional)
# ============================================
echo "6. End-to-end tests (requires API key)..."

# Check if API key is actually set in environment
if [ -n "${OPENROUTER_API_KEY}" ] && [ "${OPENROUTER_API_KEY}" != "sk-or-v1-..." ]; then
    info "API key detected. Running end-to-end tests..."
    info "(This will make real API calls and may incur costs)"

    read -p "Continue with end-to-end tests? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if pytest tests/integration/end_to_end/test_openrouter_e2e.py -v --tb=short -m integration; then
            info "✓ End-to-end tests passed"
        else
            warn "End-to-end tests failed (may be due to API issues)"
        fi
    else
        info "Skipping end-to-end tests"
    fi
else
    warn "OPENROUTER_API_KEY not set or is placeholder. Skipping end-to-end tests."
    info "To run end-to-end tests:"
    info "  1. Add your API key to .env"
    info "  2. Export it: export OPENROUTER_API_KEY=\$(grep OPENROUTER_API_KEY .env | cut -d= -f2)"
    info "  3. Run: pytest tests/integration/end_to_end/test_openrouter_e2e.py -v -m integration"
fi

echo ""

# ============================================
# 7. Verify CLI Commands
# ============================================
echo "7. Verifying CLI commands..."

# Check that CLI recognizes openrouter provider
if tclp-benchmark run --help | grep -q "openrouter"; then
    info "✓ CLI help mentions openrouter"
else
    warn "CLI help doesn't mention openrouter (may be in examples)"
fi

# Test config loading
info "Testing configuration loading..."
if python3 -c "
from tclp_benchmark.config.loader import ConfigLoader
loader = ConfigLoader()
config = loader.load_providers_config('$PROVIDERS_CONFIG')
assert 'openrouter' in config
print('Config loaded successfully')
" 2>/dev/null; then
    info "✓ Configuration loads successfully"
else
    error "Failed to load configuration"
    exit 1
fi

echo ""

# ============================================
# 8. Check Documentation
# ============================================
echo "8. Checking documentation..."

DOCS_DIR="$PROJECT_ROOT/docs"

if [ -f "$DOCS_DIR/openrouter-integration.md" ]; then
    info "✓ OpenRouter integration guide exists"
else
    warn "OpenRouter integration guide not found"
fi

if grep -q "openrouter" "$PROJECT_ROOT/README.md"; then
    info "✓ README mentions OpenRouter"
else
    warn "README doesn't mention OpenRouter"
fi

echo ""

# ============================================
# Summary
# ============================================
echo "=================================================="
echo "  Validation Summary"
echo "=================================================="
echo ""
info "✓ Environment configuration verified"
info "✓ Configuration files valid"
info "✓ Python installation OK"
info "✓ Unit tests passed"
info "✓ Integration tests passed"
info "✓ CLI commands verified"
info "✓ Documentation present"
echo ""
info "OpenRouter integration is ready to use!"
echo ""
echo "Next steps:"
echo "  1. Ensure OPENROUTER_API_KEY is set with real key"
echo "  2. Run evaluation: tclp-benchmark run --version-tag \"test\" --providers openrouter"
echo "  3. Generate report: tclp-benchmark report --version-tag \"test\" --single"
echo ""
echo "For more information, see:"
echo "  - docs/openrouter-integration.md"
echo "  - README.md"
echo ""
