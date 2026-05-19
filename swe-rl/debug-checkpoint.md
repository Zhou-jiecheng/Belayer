按照以下workflow进行debug：

那我们先debug 基础功能，首先取出export/swe_rollouts_profile_20260325_093408文件夹下的trajectory中的命令执行序列，
比如，export/swe_rollouts_profile_20260325_093408/getmoto__moto-4860__g3__i13__1774431718928799494/traj.json中"step_debug"的值。

然后参考swe-rl/generate_with_swe_remote.py编写脚本，不需要LLM生成，直接使用收集到的命令执行环境侧的轨迹复现，启动对应的容器，然后执行。这里swe pool的启动，参考swe-rl/scripts/run_swe_rl.sh 272-284行，swe server我会启动到远程的机器上。

最后在脚本的基础上，测试 checkpoint，rerun，gc 等功能。关于 checkpoint 策略的性能测试，放到后面再说。