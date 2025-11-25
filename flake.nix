{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixpkgs-unstable";
    flakelight.url = "github:nix-community/flakelight";
    flakelight.inputs.nixpkgs.follows = "nixpkgs";

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

    # uv2nix_hammer_overrides.url = "github:TyberiusPrime/uv2nix_hammer_overrides";
  };

  outputs = inputs @ {
    self,
    nixpkgs,
    flakelight,
    uv2nix,
    pyproject-nix,
    pyproject-build-systems,
    # uv2nix_hammer_overrides,
  }:
    flakelight ./. {
      inherit inputs;
      pname = "asl-research";
      systems = [
        "x86_64-linux"
        "aarch64-darwin"
      ];
      formatters = pkgs: let
        prettier = "${pkgs.nodePackages.prettier}/bin/prettier --write .";
        alejandra = "${pkgs.alejandra}/bin/alejandra .";
        just = "${pkgs.just}/bin/just --fmt --unstable";
        ruff = "${pkgs.ruff}/bin/ruff format -n .";
      in {
        "justfile" = just;
        "*.nix" = alejandra;
        "*.js" = prettier;
        "*.ts" = prettier;
        "*.jsx" = prettier;
        "*.tsx" = prettier;
        "*.md" = prettier;
        "*.json" = prettier;
        "*.astro" = prettier;
        "*.py" = ruff;
      };
      nixpkgs.config = {
        cudaSupport = false;
        allowUnfree = true;
      };
    };
}
