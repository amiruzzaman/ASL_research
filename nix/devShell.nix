{
  pkgs,
  lib,
  outputs,
}: let
  editableOverlay = outputs.lib.workspace.mkEditablePyprojectOverlay {
    root = "$REPO_ROOT";
  };
  editablePythonSet = (outputs.lib.pythonSetForPkgs pkgs).overrideScope (lib.composeManyExtensions [
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
  virtualenv = editablePythonSet.mkVirtualEnv "hello-world-dev-env" outputs.lib.workspace.deps.all;
in
  pkgs.mkShell {
    packages = with pkgs; [
      nodejs_23
      uv
      virtualenv
      mprocs
      opencv
      ffmpeg-full
      ruff
      nodePackages.prettier
      alejandra
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
  }
