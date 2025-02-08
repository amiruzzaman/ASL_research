_default:
    @just --list --unsorted --justfile {{justfile()}}

# Run the development server in mprocs
dev:
    nix develop --command nix run .#_dev

# Run a development shell
shell:
    nix develop

# Execute a single command within the shell
run *CMD:
    nix develop --command {{CMD}}

# Run an npm command within web-frontend
npm *CMD:
    cd web-frontend; nix develop --command npm {{CMD}}

# Run a uv command for python dependency management
uv *CMD:
    nix develop --command uv {{CMD}}

