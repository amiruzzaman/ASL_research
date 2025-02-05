{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixpkgs-unstable";
  };

  outputs = {
    self,
    nixpkgs,
  }: let
    forAllSystems = nixpkgs.lib.genAttrs nixpkgs.lib.systems.flakeExposed;
    pkgsFor = system:
      (import nixpkgs) {
        inherit system;
      };
  in {
    formatter = forAllSystems (system: (pkgsFor system).alejandra);
    devShells = forAllSystems (system: let
      pkgs = pkgsFor system;
    in {
      default = pkgs.mkShell {
        buildInputs = with pkgs; [
          nodejs_23
          ffmpeg-full
          (python312.withPackages (p:
            with p; [
              numpy
              imageio
              msgpack
              flask
              av
              python-dotenv
              opencv-python
              (pkgs.callPackage ./nix/mediapipe.nix {})
            ]))
        ];
      };
    });
  };
}
