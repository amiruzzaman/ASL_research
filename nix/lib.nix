{
  inputs,
  outputs,
  ...
}: let
  workspace = inputs.uv2nix.lib.workspace.loadWorkspace {workspaceRoot = ../.;};
  overlay = workspace.mkPyprojectOverlay {
    sourcePreference = "wheel";
  };
  hacksForPkgs = pkgs: pkgs.callPackage inputs.pyproject-nix.build.hacks {};
  selectPy = pkgs: pkgs.python312;

  pyOverride = pkgs: let
    hacks = hacksForPkgs pkgs;
  in
    pkgs.lib.composeExtensions overlay (_final: prev: {
      # mediapipe = hacks.nixpkgsPrebuilt {
      #   from = pkgs.mediapipe;
      #   prev = prev.mediapipe;
      # };
      torch = hacks.nixpkgsPrebuilt {
        from = pkgs.python312Packages.torchWithoutCuda;
        prev = prev.torch.overrideAttrs (old: {
          passthru =
            old.passthru
            // {
              dependencies = pkgs.lib.filterAttrs (name: _: ! pkgs.lib.hasPrefix "nvidia" name) old.passthru.dependencies;
            };
        });
      };
    });
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
        (pyOverride pkgs)
      ]
    );
}
