{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixpkgs-unstable";
    flakelight.url = "github:nix-community/flakelight";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = inputs @ {
    self,
    nixpkgs,
    flakelight,
    uv2nix,
    pyproject-nix,
    pyproject-build-systems,
    ...
  }:
    flakelight ./. (let
      workspace = uv2nix.lib.workspace.loadWorkspace {workspaceRoot = ./.;};
      overlay = workspace.mkPyprojectOverlay {
        sourcePreference = "wheel";
      };
      selectPy = pkgs: pkgs.python312;
      pythonSetForPkgs = pkgs:
        (pkgs.callPackage pyproject-nix.build.packages {
          python = selectPy pkgs;
        })
        .overrideScope
        (
          pkgs.lib.composeManyExtensions [
            pyproject-build-systems.overlays.default
            overlay
          ]
        );
    in {
      inherit inputs;
      systems = ["x86_64-linux" "x86_64-darwin" "aarch64-darwin"];
      packages = {
        backend-env = {pkgs}: (pythonSetForPkgs pkgs).mkVirtualEnv "asl-research-env" workspace.deps.default;
        frontend = {pkgs}:
          pkgs.buildNpmPackage {
            name = "asl-research-frontend";
            version = "0.1.0";
            src = ./web-frontend;
            packageJSON = ./web-frontend/package.json;
            npmDeps = pkgs.importNpmLock {
              npmRoot = ./web-frontend;
            };
            npmConfigHook = pkgs.importNpmLock.npmConfigHook;
            installPhase = "cp -r dist/ $out";
            ASTRO_TELEMETRY_DISABLED = 1;
          };
        backend = {
          pkgs,
          lib,
          outputs',
        }:
        # TODO: Setting PYTHONPATH directly feels a bit icky, does uv2nix have smth to give me the path directly?
          pkgs.writeShellApplication {
            name = "asl-research-backend";
            runtimeInputs = with pkgs; [outputs'.packages.backend-env opencv ffmpeg-full];
            text = ''
              #!/usr/bin/env sh
              export PYTHONPATH="${outputs'.packages.backend-env}/lib/python3.12/site-packages"
              ${lib.getExe pkgs.python312Packages.gunicorn} web_backend:app "$@"
            '';
          };
        _dev = {
          pkgs,
          lib,
        }:
          pkgs.writeScriptBin "run-dev-servers" ''
            #!/usr/bin/env sh
            ${lib.getExe pkgs.mprocs} --names "Backend,Frontend" "uv run dev" "cd web-frontend; npm run dev --port 4321"
          '';
      };
      devShell = {
        pkgs,
        lib,
      }: (let
        editableOverlay = workspace.mkEditablePyprojectOverlay {
          root = "$REPO_ROOT";
        };
        editablePythonSet = (pythonSetForPkgs pkgs).overrideScope (lib.composeManyExtensions [
          editableOverlay
          (final: prev: {
            asl-research = prev.asl-research.overrideAttrs (old: {
              src = lib.fileset.toSource {
                root = old.src;
                fileset = lib.fileset.unions [
                  (old.src + "/pyproject.toml")
                  (old.src + "/src")
                ];
              };
              nativeBuildInputs =
                old.nativeBuildInputs
                ++ final.resolveBuildSystem {
                  editables = [];
                };
            });
          })
        ]);
        virtualenv = editablePythonSet.mkVirtualEnv "hello-world-dev-env" workspace.deps.all;
      in
        pkgs.mkShell {
          packages = with pkgs; [
            nodejs_23
            uv
            virtualenv
            mprocs
            opencv
            ffmpeg-full
          ];

          env = {
            UV_NO_SYNC = "1";
            UV_PYTHON = "${virtualenv}/bin/python";
            UV_PYTHON_DOWNLOADS = "never";
          };

          FLASK_DEBUG = "1";
          FLASK_PORT = "5000";
          CORS_ORIGIN_ALLOW = "http://localhost:4321";
          PUBLIC_BACKEND_HOST = "http://localhost:5000";

          shellHook = ''
            unset PYTHONPATH
            export REPO_ROOT=$(git rev-parse --show-toplevel)
          '';
        });
    });
}
