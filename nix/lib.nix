{inputs, ...}: let
  workspace = inputs.uv2nix.lib.workspace.loadWorkspace {workspaceRoot = ../.;};
  overlay = workspace.mkPyprojectOverlay {
    sourcePreference = "wheel";
  };
  hacksForPkgs = pkgs: pkgs.callPackage inputs.pyproject-nix.build.hacks {};
  selectPy = pkgs: pkgs.python312;

  hammerOverride = pkgs: pkgs.lib.composeExtensions (inputs.uv2nix_hammer_overrides.overrides pkgs) overlay;

  pyOverride = pkgs: let
    hacks = hacksForPkgs pkgs;
  in
    pkgs.lib.composeExtensions (hammerOverride pkgs) (_final: prev: {
      opencv-contrib-python = hacks.nixpkgsPrebuilt {
        from = pkgs.python3Packages.opencv4;
        prev = prev.opencv-contrib-python;
      };
      torch = hacks.nixpkgsPrebuilt {
        from = pkgs.python3Packages.torchWithoutCuda;
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
            darwinSdkVersion = "12.0";
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
