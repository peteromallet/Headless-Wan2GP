Hi there,

I've been hitting a weird issue that only seems to happen on EUR-IS-1 pods.

When I spin up a pod, `nvidia-smi` works fine initially. But after installing some Python ML packages and running a workload, it starts failing with:

```
Failed to initialize NVML: Unknown Error
```

I've tried this a few times now and it keeps happening. The strange thing is I ran the exact same setup on a US region pod and it worked perfectly — no NVML issues at all.

Is there something different about the EUR-IS-1 hosts? Maybe a driver version difference or something with the container setup there?

Not urgent since I can just use other regions for now, but figured I'd flag it in case it's affecting others too.

Thanks!
