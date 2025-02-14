{
  pkgs,
  lib,
}:
# TODO: Setting PYTHONPATH directly feels a bit icky, does uv2nix have smth to give me the path directly?
pkgs.writeShellApplication {
  name = "asl-research-backend";
  runtimeInputs = with pkgs; [opencv ffmpeg-full];
  text = ''
    #!/usr/bin/env sh
    export PYTHONPATH="${pkgs.backend-env}/lib/python3.12/site-packages"
    ${lib.getExe pkgs.python312Packages.gunicorn} web_backend:app "$@"
  '';
}
