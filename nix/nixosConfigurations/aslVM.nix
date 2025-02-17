{outputs, ...}: let
  # text/html included by default
  compressMimeTypes = [
    "application/atom+xml"
    "application/geo+json"
    "application/javascript"
    "application/json"
    "application/ld+json"
    "application/manifest+json"
    "application/rdf+xml"
    "application/vnd.ms-fontobject"
    "application/wasm"
    "application/x-rss+xml"
    "application/x-web-app-manifest+json"
    "application/xhtml+xml"
    "application/xliff+xml"
    "application/xml"
    "application/x-msgpack"
    "font/collection"
    "font/otf"
    "font/ttf"
    "image/bmp"
    "image/svg+xml"
    "image/vnd.microsoft.icon"
    "text/cache-manifest"
    "text/calendar"
    "text/css"
    "text/csv"
    "text/javascript"
    "text/markdown"
    "text/plain"
    "text/vcard"
    "text/vnd.rim.location.xloc"
    "text/vtt"
    "text/x-component"
    "text/xml"
  ];
in {
  system = "x86_64-linux";

  modules = [
    outputs.nixosModules.aslSite
    (
      {
        outputs',
        pkgs,
        modulesPath,
        lib,
        ...
      }: {
        imports = ["${modulesPath}/virtualisation/qemu-vm.nix"];

        virtualisation = {
          graphics = false;
          diskImage = null;
          sharedDirectories.words = {
            source = "$WORD_DIR";
            target = "/word-mnt";
          };
          forwardPorts = [
            {
              from = "host";
              guest.port = 8080;
              host.port = 8080;
            }
            {
              from = "host";
              guest.port = 8443;
              host.port = 8443;
            }
          ];
        };

        users = {
          mutableUsers = false;
          users."root" = {
            hashedPassword = "$y$j9T$sfyYVDmioHNdul1/euqAQ1$JFPf2l70Nw.rfH2ku7kr5oHZebJoRm9UDiWX4g3v7k9";
          };
        };

        system.stateVersion = "25.05";
        time.timeZone = "America/New_York";

        services.aslSite = {
          enable = true;
          wordDir = "/word-mnt";
        };

        networking.firewall.allowedTCPPorts = [8080 8443];

        services.nginx = {
          enable = true;
          recommendedOptimisation = true;
          # recommendedBrotliSettings doesn't include application/x-msgpack
          # and sadly doesn't expose compressMimeTypes, so we'll set what it
          # does manually
          additionalModules = with pkgs.nginxModules; [brotli];
          appendHttpConfig = ''
            brotli on;
            brotli_static on;
            brotli_comp_level 5;
            brotli_window 512k;
            brotli_min_length 256;
            brotli_types ${lib.concatStringsSep " " compressMimeTypes};
          '';
          virtualHosts.aslResearch = {
            listen = [
              {
                addr = "0.0.0.0";
                port = 8080;
              }
              {
                addr = "0.0.0.0";
                port = 8443;
              }
            ];
            default = true;
            root = "${pkgs.frontend}";
            locations."/_astro".extraConfig = ''
              expires 604800;
            '';
            locations."/api" = {
              recommendedProxySettings = true;
              proxyPass = "http://127.0.0.1:8000";
            };
          };
        };
      }
    )
  ];
}
