{
  lib,
  pkgs,
  config,
  ...
}: let
  cfg = config.services.aslSite;
in {
  options.services.aslSite = {
    enable = lib.mkEnableOption "ASL <-> ENG Web Application";

    wordDir = lib.mkOption {
      default = "/var/lib/asl-site/words";
      type = lib.types.path;
      description = ''
        Path to store and read known ASL Gloss animation files.

        This directory should contain a set of files following the pattern of WORD.msgpack,
        WORD will be what gloss term the animation is performing.
      '';
    };

    address = lib.mkOption {
      default = "127.0.0.1";
      type = lib.types.str;
      description = ''
        Address to bind to, you'll most likely want to put this service behind a reverse
        proxy if you want to expose it to the outside world, so by default this is only
        the loopback address.
      '';
    };

    port = lib.mkOption {
      default = 8000;
      type = lib.types.port;
      description = ''
        Port to bind to, automatically opened in the firewall if `openFirewall` is true.
      '';
    };

    openFirewall = lib.mkOption {
      default = false;
      type = lib.types.bool;
      description = ''
        Whether to open the firewall for this application. Please note it's recommended
        you put this app behind a reverse-proxy, therefore this option is off by default.
      '';
    };

    extraArgs = lib.mkOption {
      default = [];
      type = lib.types.listOf lib.types.str;
      example = lib.literalExpression ''
        [ "--threads 4" ]
      '';
      description = ''
        Extra arguments to pass to the web server.
        See [the Gunicorn documentation](https://docs.gunicorn.org/en/stable/run.html#commonly-used-arguments) for options.
      '';
    };
  };

  config = lib.mkIf cfg.enable {
    systemd.services.asl-site = {
      description = "ASL Web API";
      after = ["network.target"];
      wantedBy = ["multi-user.target"];
      serviceConfig.ExecStart = ''
        ${lib.getExe pkgs.backend} -e WORD_LIBRARY_LOCATION=${cfg.wordDir} --bind ${cfg.address}:${builtins.toString cfg.port} ${lib.concatStringsSep " " cfg.extraArgs}
      '';
    };

    networking.firewall.allowedTCPPorts = lib.mkIf cfg.openFirewall [cfg.port];
  };
}
