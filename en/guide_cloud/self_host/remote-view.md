# Remote Access to Offline Dashboard

The `swanlab watch` command makes offline access to experiments very simple, and in machine learning training, using remote servers is quite common.

This section will teach you:

- How to set the IP and port for the experiment dashboard.
- How to access the experiment dashboard from your local machine.

## Preparations

- `Note down the remote IP`: For example, if you are using a cloud server, its public IP (in the form of 8.141.192.68) is the IP you will use to access the experiment dashboard from your local machine; if you are using a LAN server, note down its LAN IP.
- `Open the port`: First, you need to check the security group/firewall of the remote server. For example, if you want the experiment dashboard to use port `5092`, you need to check if the server has opened this port.

> You can use the `telnet <server IP> <port number>` command to check if the port on the Linux server is open.

## Set the IP and Port for the Experiment Dashboard on the Remote Server

We need to run the experiment dashboard service on the remote server (the machine where the training is running).

In the `swanlab watch` command, the main parameters that can be set are `-p` and `-h`:

| API         | Description                                     | Example                                             |
|-------------|-------------------------------------------------|-----------------------------------------------------|
| `-p, --port`| Set the port for the experiment dashboard web service, default is 5092. | `swanlab watch -p 8080`: Set the experiment dashboard web service to port 8080 |
| `-h, --host`| Set the IP address for the experiment dashboard web service, default is 127.0.0.1. | `swanlab watch -h 0.0.0.0`: Set the experiment dashboard web service IP address to 0.0.0.0 |

Generally, for remote access to the experiment dashboard, you need to set `-h` to `0.0.0.0`, and `-p` according to your needs. Here, we set the port to `8080`:

```shell
swanlab watch -h 0.0.0.0 -p 8080
```

Running the above command gives:

![image](/assets/self-host_im.jpg)

## Access the Experiment Dashboard from the Local Machine

Now, on your local machine, open a browser and access `remote IP address:port number`.

For example, if my remote server's public IP is `8.146.xxx.71` and the port number is set to `8080`, then in the browser, I access `8.146.xxx.71:8080`.