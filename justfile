_default:
    @just --list --unsorted --justfile {{justfile()}}

# Run the development server in mprocs
dev:
    just run "mprocs --names 'Backend,Frontend' 'python src/web_backend/__init__.py' 'cd web-frontend; npm run dev'"

# Run a development shell
shell:
    nix develop

# Execute a single command within the shell
run *CMD:
    nix develop --command {{CMD}}

# Run an npm command within web-frontend
npm *CMD:
    cd web-frontend; just run npm {{CMD}}

# Run a uv command for python dependency management
uv *CMD:
    just run uv {{CMD}}

