_default:
    @just --list --unsorted --justfile {{justfile()}}

# Run the development server in mprocs
dev:
    just run "mprocs --names 'Frontend,Backend' 'cd web-frontend; npm run dev' 'python src/web_backend/__init__.py'"

# Run a development shell
shell:
    nix develop

# Execute a single command within the shell
run *CMD:
    nix develop --command {{CMD}}

# Run an npm command within web-frontend
[working-directory: 'web-frontend']
npm *CMD:
    nix develop --command npm {{CMD}}

# Run a uv command for python dependency management
uv *CMD:
    nix develop --command uv {{CMD}}

# Run a production VM site
vm $PROJ_ROOT=`echo $PWD`:
    mkdir -p words
    nix run .#nixosConfigurations.aslVM.config.system.build.vm

