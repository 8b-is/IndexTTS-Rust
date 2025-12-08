#!/usr/bin/env bash
# ============================================================================
# IndexTTS-Rust Management Script
# ============================================================================
# A colorful, wonderful management script for our TTS engine!
# Built with love by Hue & Aye for Trisha in Accounting 💜
#
# Usage: ./scripts/manage.sh [command] [options]
# Run without arguments to see the interactive menu
# ============================================================================

set -e

# ============================================================================
# 🎨 COLORS & STYLING - Because life is too short for boring terminals!
# ============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'

# Fancy symbols (because we're fancy like that)
CHECK="${GREEN}✓${RESET}"
CROSS="${RED}✗${RESET}"
ARROW="${CYAN}➜${RESET}"
STAR="${YELLOW}★${RESET}"
SPARKLE="${MAGENTA}✨${RESET}"
ROCKET="${CYAN}🚀${RESET}"
GEAR="${BLUE}⚙${RESET}"
BROOM="${YELLOW}🧹${RESET}"
DOCKER="${BLUE}🐳${RESET}"
TEST="${GREEN}🧪${RESET}"
WARN="${YELLOW}⚠${RESET}"

# ============================================================================
# 🎭 BANNER - First impressions matter!
# ============================================================================
print_banner() {
    echo -e "${CYAN}"
    cat << 'EOF'
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║   ██╗███╗   ██╗██████╗ ███████╗██╗  ██╗████████╗████████╗███████╗║
    ║   ██║████╗  ██║██╔══██╗██╔════╝╚██╗██╔╝╚══██╔══╝╚══██╔══╝██╔════╝║
    ║   ██║██╔██╗ ██║██║  ██║█████╗   ╚███╔╝    ██║      ██║   ███████╗║
    ║   ██║██║╚██╗██║██║  ██║██╔══╝   ██╔██╗    ██║      ██║   ╚════██║║
    ║   ██║██║ ╚████║██████╔╝███████╗██╔╝ ██╗   ██║      ██║   ███████║║
    ║   ╚═╝╚═╝  ╚═══╝╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝      ╚═╝   ╚══════╝║
    ║                                                                  ║
    ║   🎤 Text-to-Speech Engine in Pure Rust 🦀                       ║
    ║   Built by Hue & Aye | Approved by Trisha in Accounting 💜       ║
    ╚══════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${RESET}"
}

# ============================================================================
# 📍 PROJECT ROOT - Find our way home
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ============================================================================
# 🔧 HELPER FUNCTIONS - The unsung heroes
# ============================================================================
log_info() {
    echo -e "${ARROW} ${WHITE}$1${RESET}"
}

log_success() {
    echo -e "${CHECK} ${GREEN}$1${RESET}"
}

log_error() {
    echo -e "${CROSS} ${RED}$1${RESET}"
}

log_warn() {
    echo -e "${WARN} ${YELLOW}$1${RESET}"
}

log_step() {
    echo -e "\n${STAR} ${BOLD}${MAGENTA}$1${RESET}"
}

# Spinner for long-running tasks (Trisha loves spinners)
spin() {
    local pid=$1
    local delay=0.1
    local spinstr='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    while ps -p $pid > /dev/null 2>&1; do
        local temp=${spinstr#?}
        printf " ${CYAN}[%c]${RESET}" "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b"
    done
    printf "     \b\b\b\b\b"
}

# Check if a command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# ============================================================================
# 🏗️ BUILD COMMANDS - Where the magic happens!
# ============================================================================
cmd_build() {
    local mode="${1:-release}"
    log_step "Building IndexTTS ($mode mode)"

    case "$mode" in
        release)
            log_info "Compiling with maximum optimizations (LTO enabled)..."
            cargo build --release 2>&1
            log_success "Release build complete! Binary at: target/release/indextts"
            ;;
        debug)
            log_info "Compiling debug build..."
            cargo build 2>&1
            log_success "Debug build complete! Binary at: target/debug/indextts"
            ;;
        *)
            log_error "Unknown build mode: $mode (use 'release' or 'debug')"
            return 1
            ;;
    esac
}

# Full pre-commit build workflow (as Aye commands!)
cmd_build_full() {
    log_step "Running Full Build Workflow ${ROCKET}"
    echo -e "${DIM}(Build → Clippy → Build again - the holy trinity!)${RESET}\n"

    log_info "Step 1/3: Initial release build..."
    cargo build --release
    log_success "Initial build complete!"

    log_info "Step 2/3: Running Clippy (the lint police)..."
    cargo clippy -- -D warnings
    log_success "Clippy is happy! No warnings detected."

    log_info "Step 3/3: Final verification build..."
    cargo build --release
    log_success "Final build complete!"

    echo -e "\n${SPARKLE} ${GREEN}${BOLD}All systems go! Ready for commit!${RESET} ${SPARKLE}"
}

# ============================================================================
# 🧪 TEST COMMANDS - Trust but verify!
# ============================================================================
cmd_test() {
    local filter="$1"
    log_step "Running Tests ${TEST}"

    if [[ -n "$filter" ]]; then
        log_info "Running tests matching: '$filter'"
        cargo test "$filter" -- --nocapture
    else
        log_info "Running all tests..."
        cargo test -- --nocapture
    fi
    log_success "All tests passed!"
}

cmd_test_integration() {
    log_step "Running Integration Tests ${TEST}"
    log_info "Looking for integration tests..."

    if [[ -d "tests" ]]; then
        cargo test --test '*' -- --nocapture
        log_success "Integration tests complete!"
    else
        log_warn "No tests/ directory found. Creating integration test structure..."
        mkdir -p tests
        log_info "Integration test directory created. Add your tests there!"
    fi
}

cmd_bench() {
    local bench_name="$1"
    log_step "Running Benchmarks 📊"

    if [[ -n "$bench_name" ]]; then
        log_info "Running benchmark: $bench_name"
        cargo bench --bench "$bench_name"
    else
        log_info "Running all benchmarks..."
        cargo bench
    fi
    log_success "Benchmarks complete! Check target/criterion for reports."
}

# ============================================================================
# 🧹 CLEAN COMMANDS - Marie Kondo would be proud!
# ============================================================================
cmd_clean() {
    log_step "Cleaning Build Artifacts ${BROOM}"

    log_info "Running cargo clean..."
    cargo clean
    log_success "Cargo artifacts cleaned!"

    # Clean any generated files
    if [[ -f "output.wav" ]]; then
        rm -f output.wav
        log_info "Removed output.wav"
    fi

    # Clean .pyc and __pycache__ if any Python remnants exist
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true

    log_success "Project is squeaky clean! ✨"
}

cmd_clean_all() {
    log_step "Deep Clean (Nuclear Option) ${BROOM}"
    log_warn "This will remove target/, models/, and all generated files!"

    read -p "Are you sure? (y/N): " confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        cargo clean
        rm -rf models/ 2>/dev/null || true
        rm -f output.wav chris_cloned.wav 2>/dev/null || true
        find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        log_success "Deep clean complete! It's like a fresh start."
    else
        log_info "Clean cancelled. Your files are safe!"
    fi
}

# ============================================================================
# 🐳 DOCKER COMMANDS - Containerize all the things!
# ============================================================================
DOCKER_IMAGE="indextts-rust"
DOCKER_TAG="latest"

cmd_docker_build() {
    log_step "Building Docker Image ${DOCKER}"

    if [[ ! -f "Dockerfile" ]]; then
        log_warn "No Dockerfile found. Creating a basic one..."
        cmd_docker_init
    fi

    log_info "Building image: ${DOCKER_IMAGE}:${DOCKER_TAG}"
    docker build -t "${DOCKER_IMAGE}:${DOCKER_TAG}" .
    log_success "Docker image built successfully!"
}

cmd_docker_run() {
    log_step "Running Docker Container ${DOCKER}"

    log_info "Starting container..."
    docker run -it --rm \
        -v "$(pwd)/examples:/app/examples" \
        -v "$(pwd)/models:/app/models" \
        "${DOCKER_IMAGE}:${DOCKER_TAG}" "$@"
}

cmd_docker_shell() {
    log_step "Opening Docker Shell ${DOCKER}"

    docker run -it --rm \
        -v "$(pwd):/app" \
        "${DOCKER_IMAGE}:${DOCKER_TAG}" /bin/bash
}

cmd_docker_init() {
    log_step "Creating Dockerfile"

    cat > Dockerfile << 'DOCKERFILE'
# IndexTTS-Rust Dockerfile
# Multi-stage build for minimal image size

# Build stage
FROM rust:1.75-bookworm AS builder

WORKDIR /app
COPY . .

# Build release binary
RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim

# Install ONNX Runtime dependencies
RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/target/release/indextts /usr/local/bin/
COPY --from=builder /app/examples /app/examples

ENTRYPOINT ["indextts"]
CMD ["--help"]
DOCKERFILE
    log_success "Dockerfile created!"
}

# ============================================================================
# 📋 LINT & FORMAT - Keep it pretty!
# ============================================================================
cmd_lint() {
    log_step "Running Linters ${GEAR}"

    log_info "Running Clippy (deny warnings mode)..."
    cargo clippy -- -D warnings
    log_success "Clippy approved! Code is lint-free."
}

cmd_format() {
    log_step "Formatting Code ${GEAR}"

    log_info "Running cargo fmt..."
    cargo fmt
    log_success "Code formatted beautifully!"
}

cmd_format_check() {
    log_step "Checking Format ${GEAR}"

    log_info "Checking if code is properly formatted..."
    if cargo fmt -- --check; then
        log_success "Code is properly formatted!"
    else
        log_error "Code needs formatting. Run: ./scripts/manage.sh format"
        return 1
    fi
}

# ============================================================================
# 📦 DEPENDENCY MANAGEMENT - Stay up to date!
# ============================================================================
cmd_deps_check() {
    log_step "Checking Dependencies 📦"

    if command_exists cargo-outdated; then
        log_info "Checking for outdated dependencies..."
        cargo outdated
    else
        log_warn "cargo-outdated not installed. Install with: cargo install cargo-outdated"
        log_info "Showing dependency tree instead..."
        cargo tree --depth 1
    fi
}

cmd_deps_update() {
    log_step "Updating Dependencies 📦"

    log_info "Updating Cargo.lock..."
    cargo update
    log_success "Dependencies updated! Remember to test after updating."
}

# ============================================================================
# 🔍 INFO & STATUS - Know thyself!
# ============================================================================
cmd_info() {
    log_step "System Information ℹ️"

    echo -e "\n${BOLD}Project:${RESET}"
    echo -e "  Name:     ${CYAN}IndexTTS-Rust${RESET}"
    echo -e "  Version:  ${CYAN}$(grep '^version' Cargo.toml | head -1 | cut -d'"' -f2)${RESET}"
    echo -e "  Edition:  ${CYAN}$(grep '^edition' Cargo.toml | cut -d'"' -f2)${RESET}"

    echo -e "\n${BOLD}Toolchain:${RESET}"
    echo -e "  Rust:     ${CYAN}$(rustc --version)${RESET}"
    echo -e "  Cargo:    ${CYAN}$(cargo --version)${RESET}"

    echo -e "\n${BOLD}Environment:${RESET}"
    echo -e "  OS:       ${CYAN}$(uname -s) $(uname -r)${RESET}"
    echo -e "  Arch:     ${CYAN}$(uname -m)${RESET}"

    if command_exists docker; then
        echo -e "  Docker:   ${CYAN}$(docker --version)${RESET}"
    else
        echo -e "  Docker:   ${DIM}Not installed${RESET}"
    fi

    # Check for ONNX Runtime
    echo -e "\n${BOLD}ONNX Runtime:${RESET}"
    if [[ -n "$ORT_DYLIB_PATH" ]]; then
        echo -e "  Path:     ${CYAN}$ORT_DYLIB_PATH${RESET}"
    else
        echo -e "  Path:     ${YELLOW}Not set (ORT_DYLIB_PATH)${RESET}"
    fi

    echo -e "\n${BOLD}Models:${RESET}"
    if [[ -d "models" ]]; then
        local model_count=$(find models -name "*.onnx" 2>/dev/null | wc -l | tr -d ' ')
        echo -e "  ONNX:     ${CYAN}${model_count} model(s) found${RESET}"
    else
        echo -e "  ONNX:     ${YELLOW}No models/ directory${RESET}"
    fi
}

cmd_check() {
    log_step "Quick Compilation Check ${GEAR}"

    log_info "Running cargo check..."
    cargo check
    log_success "Compilation check passed!"
}

# ============================================================================
# 🎤 TTS COMMANDS - Let's make some noise!
# ============================================================================
cmd_synthesize() {
    local text="$1"
    local voice="${2:-examples/voice_01.wav}"
    local output="${3:-output.wav}"

    log_step "Synthesizing Speech 🎤"

    if [[ -z "$text" ]]; then
        log_error "Usage: ./manage.sh synthesize \"Your text here\" [voice.wav] [output.wav]"
        return 1
    fi

    log_info "Text: \"$text\""
    log_info "Voice: $voice"
    log_info "Output: $output"

    ./target/release/indextts synthesize \
        --text "$text" \
        --voice "$voice" \
        --output "$output"

    log_success "Audio saved to: $output"
}

# ============================================================================
# 🌍 ENVIRONMENT MANAGEMENT - Set the stage!
# ============================================================================
cmd_env_setup() {
    log_step "Setting Up Environment 🌍"

    log_info "Creating .env template..."
    cat > .env.example << 'ENVFILE'
# IndexTTS-Rust Environment Variables
# Copy this to .env and customize

# ONNX Runtime library path (required)
# ORT_DYLIB_PATH=/path/to/libonnxruntime.so

# Logging level (error, warn, info, debug, trace)
RUST_LOG=info

# Number of threads for parallel processing (0 = auto)
# RAYON_NUM_THREADS=0
ENVFILE

    if [[ ! -f ".env" ]]; then
        cp .env.example .env
        log_success ".env file created from template!"
    else
        log_info ".env already exists. Check .env.example for new options."
    fi

    # Ensure .env is in .gitignore
    if ! grep -q "^\.env$" .gitignore 2>/dev/null; then
        echo ".env" >> .gitignore
        log_info "Added .env to .gitignore"
    fi

    log_success "Environment setup complete!"
}

cmd_env_show() {
    log_step "Current Environment 🌍"

    echo -e "\n${BOLD}Relevant Environment Variables:${RESET}"
    echo -e "  RUST_LOG:           ${CYAN}${RUST_LOG:-<not set>}${RESET}"
    echo -e "  ORT_DYLIB_PATH:     ${CYAN}${ORT_DYLIB_PATH:-<not set>}${RESET}"
    echo -e "  RAYON_NUM_THREADS:  ${CYAN}${RAYON_NUM_THREADS:-<not set>}${RESET}"

    if [[ -f ".env" ]]; then
        echo -e "\n${BOLD}Contents of .env:${RESET}"
        grep -v "^#" .env | grep -v "^$" | while read line; do
            echo -e "  ${DIM}$line${RESET}"
        done
    fi
}

# ============================================================================
# 📚 HELP & MENU - Because we all need a little guidance
# ============================================================================
cmd_help() {
    print_banner
    echo -e "${BOLD}USAGE:${RESET}"
    echo -e "  ./scripts/manage.sh ${CYAN}<command>${RESET} [options]"
    echo -e ""
    echo -e "${BOLD}BUILD COMMANDS:${RESET}"
    echo -e "  ${CYAN}build${RESET} [release|debug]  Build the project (default: release)"
    echo -e "  ${CYAN}build-full${RESET}             Full workflow: Build → Clippy → Build"
    echo -e "  ${CYAN}check${RESET}                  Quick compilation check (no codegen)"
    echo -e ""
    echo -e "${BOLD}TEST COMMANDS:${RESET}"
    echo -e "  ${CYAN}test${RESET} [filter]          Run tests (optional filter)"
    echo -e "  ${CYAN}test-integration${RESET}       Run integration tests"
    echo -e "  ${CYAN}bench${RESET} [name]           Run benchmarks"
    echo -e ""
    echo -e "${BOLD}CODE QUALITY:${RESET}"
    echo -e "  ${CYAN}lint${RESET}                   Run Clippy linter"
    echo -e "  ${CYAN}format${RESET}                 Format code with rustfmt"
    echo -e "  ${CYAN}format-check${RESET}           Check if code is formatted"
    echo -e ""
    echo -e "${BOLD}CLEAN COMMANDS:${RESET}"
    echo -e "  ${CYAN}clean${RESET}                  Clean build artifacts"
    echo -e "  ${CYAN}clean-all${RESET}              Deep clean (includes models/)"
    echo -e ""
    echo -e "${BOLD}DOCKER COMMANDS:${RESET}"
    echo -e "  ${CYAN}docker-build${RESET}           Build Docker image"
    echo -e "  ${CYAN}docker-run${RESET}             Run container"
    echo -e "  ${CYAN}docker-shell${RESET}           Open shell in container"
    echo -e "  ${CYAN}docker-init${RESET}            Create Dockerfile"
    echo -e ""
    echo -e "${BOLD}DEPENDENCIES:${RESET}"
    echo -e "  ${CYAN}deps-check${RESET}             Check for outdated deps"
    echo -e "  ${CYAN}deps-update${RESET}            Update dependencies"
    echo -e ""
    echo -e "${BOLD}ENVIRONMENT:${RESET}"
    echo -e "  ${CYAN}env-setup${RESET}              Set up .env file"
    echo -e "  ${CYAN}env-show${RESET}               Show current environment"
    echo -e ""
    echo -e "${BOLD}TTS:${RESET}"
    echo -e "  ${CYAN}synthesize${RESET} \"text\"      Synthesize speech"
    echo -e ""
    echo -e "${BOLD}INFO:${RESET}"
    echo -e "  ${CYAN}info${RESET}                   Show system information"
    echo -e "  ${CYAN}help${RESET}                   Show this help message"
    echo -e ""
    echo -e "${DIM}Made with 💜 by Hue & Aye for the IndexTTS project${RESET}"
}

# Interactive menu when no command is given
interactive_menu() {
    print_banner
    echo -e "${BOLD}What would you like to do today?${RESET}\n"

    PS3=$'\n'"${ARROW} Select an option: "

    options=(
        "🏗️  Build (Release)"
        "🏗️  Build (Full Workflow)"
        "🧪  Run Tests"
        "📊  Run Benchmarks"
        "🔍  Lint (Clippy)"
        "🎨  Format Code"
        "🧹  Clean"
        "🐳  Docker Build"
        "ℹ️   System Info"
        "❓  Help"
        "🚪  Exit"
    )

    select opt in "${options[@]}"; do
        case $REPLY in
            1) cmd_build release ;;
            2) cmd_build_full ;;
            3) cmd_test ;;
            4) cmd_bench ;;
            5) cmd_lint ;;
            6) cmd_format ;;
            7) cmd_clean ;;
            8) cmd_docker_build ;;
            9) cmd_info ;;
            10) cmd_help ;;
            11) echo -e "\n${SPARKLE} See you next time! ${SPARKLE}"; exit 0 ;;
            *) echo -e "${CROSS} Invalid option. Try again!" ;;
        esac
        echo ""
        read -p "Press Enter to continue..."
        interactive_menu
        break
    done
}

# ============================================================================
# 🚀 MAIN ENTRY POINT - Here we go!
# ============================================================================
main() {
    local command="${1:-}"
    shift 2>/dev/null || true

    case "$command" in
        # Build commands
        build)          cmd_build "$@" ;;
        build-full)     cmd_build_full ;;
        check)          cmd_check ;;

        # Test commands
        test)           cmd_test "$@" ;;
        test-integration) cmd_test_integration ;;
        bench)          cmd_bench "$@" ;;

        # Code quality
        lint)           cmd_lint ;;
        format)         cmd_format ;;
        format-check)   cmd_format_check ;;

        # Clean commands
        clean)          cmd_clean ;;
        clean-all)      cmd_clean_all ;;

        # Docker commands
        docker-build)   cmd_docker_build ;;
        docker-run)     cmd_docker_run "$@" ;;
        docker-shell)   cmd_docker_shell ;;
        docker-init)    cmd_docker_init ;;

        # Dependencies
        deps-check)     cmd_deps_check ;;
        deps-update)    cmd_deps_update ;;

        # Environment
        env-setup)      cmd_env_setup ;;
        env-show)       cmd_env_show ;;

        # TTS
        synthesize)     cmd_synthesize "$@" ;;

        # Info
        info)           cmd_info ;;
        help|--help|-h) cmd_help ;;

        # Interactive menu
        "")             interactive_menu ;;

        *)
            log_error "Unknown command: $command"
            echo -e "Run ${CYAN}./scripts/manage.sh help${RESET} for usage."
            exit 1
            ;;
    esac
}

# Let's rock! 🎸
main "$@"
