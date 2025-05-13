# FAQ

## How to change the port?

The self-hosted version of SwanLab is deployed based on [Docker](https://www.docker.com/). By default, it uses port `8000`. Modifying the default access port of the self-hosted service actually means modifying the mapping port of the **swanlab-traefik** container. There are two scenarios:

### Modifying before deployment

The installation script provides some configuration options, including data storage location and mapped port. We can modify the port by changing the script startup parameters.

- After executing the `install.sh` installation script, the command line will prompt for configuration options, where you can enter the corresponding parameters interactively. When the command line outputs `2. Use the default port (8000)? (y/n):`, enter `n`, and then you will be prompted with `Enter a custom port:`. Enter the desired port number, for example, `80`.

```bash
❯ bash install.sh
🤩 Docker is installed, so let's get started.
🧐 Checking if Docker is running...

1. Use the default path  (./data)? (y/n):
   The selected path is: ./data
2. Use the default port  (8000)? (y/n):
```

- Add parameters when running the script. The installation script provides a command-line parameter `-p` that can be used to modify the port, for example: `./install.sh -p 80`.

> For more command-line parameters, see:  [Deploy via Docker](https://github.com/SwanHubX/self-hosted/tree/main/docker)

### Modifying after deployment

If you need to modify the access port after the SwanLab service has been deployed, you need to modify the generated `docker-compose.yaml` configuration file.

Find the `swanlab/` directory at the location where the script was executed, run `cd swanlab/` to enter the `swanlab` directory, locate the corresponding `docker-compose.yaml` configuration file, and then modify the port `ports` for the `traefik` container as shown below:

```yaml
traefik:
  <<: *common
  image: ccr.ccs.tencentyun.com/self-hosted/traefik:v3.0
  container_name: swanlab-traefik
  ports:
    - "8000:80" # [!code --]
    - "80:80" # [!code ++]
```

> The above changes the access port to `80`

After making the changes, execute `docker compose up -d` to restart the container. Once restarted, you can access it via `http://{ip}:80`
