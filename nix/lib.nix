{
  inputs,
  outputs,
  ...
}: let
  workspace = inputs.uv2nix.lib.workspace.loadWorkspace {workspaceRoot = ../.;};
  overlay = workspace.mkPyprojectOverlay {
    sourcePreference = "wheel";
  };
  hacks = pkgs: pkgs.callPackage inputs.pyproject-nix.build.hacks {};
  selectPy = pkgs: pkgs.python312;
  pyOverride = pkgs: _final: prev: {
    mediapipe = (hacks pkgs).nixpkgsPrebuilt {
      from = outputs.packages.${pkgs.system}.mediapipe;
      prev = prev.mediapipe;
    };
  };
in {
  inherit workspace;
  pythonSetForPkgs = pkgs:
    (pkgs.callPackage inputs.pyproject-nix.build.packages {
      python = selectPy pkgs;
      stdenv = pkgs.stdenv.override {
        targetPlatform =
          pkgs.stdenv.targetPlatform
          // {
            darwinSdkVersion = "10.7";
          };
      };
    })
    .overrideScope
    (
      pkgs.lib.composeManyExtensions [
        inputs.pyproject-build-systems.overlays.default
        overlay
        (pyOverride pkgs)
        (_final: prev: {pythonPkgsBuildHost = prev.pythonPkgsBuildHost.overrideScope (pyOverride pkgs);})
      ]
    );
}
