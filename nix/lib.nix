{
  inputs,
  lib,
  ...
}: let
  src = lib.fileset.toSource {
    root = ../.;
    fileset = lib.fileset.unions [
      ../uv.lock
      ../pyproject.toml
      ../src
    ];
  };

  workspace = inputs.uv2nix.lib.workspace.loadWorkspace {workspaceRoot = src.outPath;};
  overlay = workspace.mkPyprojectOverlay {
    sourcePreference = "wheel";
  };
  hacksForPkgs = pkgs: pkgs.callPackage inputs.pyproject-nix.build.hacks {};
  selectPy = pkgs: pkgs.python312;

  # hammerOverride = pkgs: pkgs.lib.composeExtensions (inputs.uv2nix_hammer_overrides.overrides pkgs) overlay;

  pyOverride = pkgs: let
    hacks = hacksForPkgs pkgs;
  in
    pkgs.lib.composeExtensions overlay (_final: prev: {
      opencv-contrib-python = hacks.nixpkgsPrebuilt {
        from = pkgs.python3Packages.opencv4;
        prev = prev.opencv-contrib-python;
      };
      mediapipe = prev.mediapipe.overrideAttrs (old: {
        buildInputs = old.buildInputs ++ [pkgs.git];
        postInstall = "cd $out/lib/python3.12/site-packages; ls; git apply < ${./holistic-null-ptr.patch}";
      });
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
