{ pkgs, ... }:
let
  task = pkgs.runCommand "holistic-task" { } ''
    mkdir -p $out
    ln -s ${pkgs.holistic-task} $out/holistic-task
  '';
in
pkgs.dockerTools.buildLayeredImage {
  name = "asl-oci";
  contents = with pkgs.dockerTools; [
    usrBinEnv
    binSh
    pkgs.hello
    pkgs.backend-env
    task
  ];
  config.Cmd = "/bin/sh";
}
