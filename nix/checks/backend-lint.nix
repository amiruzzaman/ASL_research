{
  pkgs,
  lib,
  ...
}: let
  ruff = lib.getExe pkgs.ruff;
  src = ./../..;
in
  pkgs.runCommand "backend-lint" {
    PYTHONPATH = "${pkgs.backend-env}/lib/python3.12/site-packages";
    RUFF_NO_CACHE = "true";
  } ''
    cd ${src};

    ${ruff} format --check
    ${ruff} check

    mkdir $out
  ''
