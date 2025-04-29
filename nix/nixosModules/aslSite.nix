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

    models = {
      englishToGloss = lib.mkOption {
        type = lib.types.path;
        description = "Path to the .pt file containing the PyTorch model for converting English to ASL Gloss.";
      };
      glossToEnglish = lib.mkOption {
        type = lib.types.path;
        description = "Path to the .pt file containing the PyTorch model for converting ASL Gloss to English";
      };
      aslToGloss = lib.mkOption {
        type = lib.types.path;
        description = "Path to the .pt file containing the PyTorch model for converting ASL to ASL Gloss";
      };
      aslSigns = lib.mkOption {
        type = lib.types.path;
        description = "Path to a folder of signs processed by the ASL processor for use in ASL to English";
      };
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
      environment = {
        WORD_LIBRARY_LOCATION = cfg.wordDir;
        ENGLISH_2_GLOSS_MODEL_PATH = cfg.models.englishToGloss;
        GLOSS_2_ENGLISH_MODEL_PATH = cfg.models.glossToEnglish;
        SIGN_2_GLOSS_MODEL_PATH = cfg.models.aslToGloss;
        ASL_2_GLOSS_SIGNS_DIR = cfg.models.aslSigns;
      };
      script = ''
        ${lib.getExe pkgs.backend} --bind ${cfg.address}:${builtins.toString cfg.port} ${lib.concatStringsSep " " cfg.extraArgs}
      '';
      serviceConfig = {
        Restart = "always";
        RestartSec = "5s";
        # User = "asl-site";
        # Group = "asl-site";
        # StateDirectory = "asl-site";
        # DynamicUser = true;
      };
    };

    networking.firewall.allowedTCPPorts = lib.mkIf cfg.openFirewall [cfg.port];
  };
}
