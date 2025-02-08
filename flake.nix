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
  }:
    flakelight ./. {
      inherit inputs;
      systems = ["x86_64-linux" "x86_64-darwin" "aarch64-darwin"];
      formatters = {
        "*.nix" = "alejandra .";
        "*.py" = "black .";
        "*.{ts,css,astro,json}" = "prettier --write web-frontend";
      };
    };
}
