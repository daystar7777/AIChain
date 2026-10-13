# 도구 코드 실행 격리안 (결정 필요)

작성: 2026-10-11 · 상태: **A안 구현·검증 완료** (`SystemdExecutor`). 기본값은 여전히 비활성이고, `--workshop --workshop-exec systemd`를 줄 때만 실행한다.

## 왜 지금은 끄는가

서버(Contabo, Ubuntu 24.04)는 다른 서비스(nginx, Python 앱)가 돌고 있고, tailscale로 사용자 Mac들과 연결돼 있다. Docker와 gVisor가 없다. 그리고 Ubuntu 24.04는 AppArmor로 일반 프로그램의 사용자 네임스페이스를 막는다(`kernel.apparmor_restrict_unprivileged_userns=1`). 그래서 bwrap이나 nsjail을 일반 사용자로 바로 쓰기 어렵다(Chromium 샌드박스도 같은 이유로 막혔다).

## 요구사항 (05 문서 2.3)

네트워크 없음, 읽기 전용 루트, 일회용 작업 디렉터리, CPU 1·메모리 ≤512MB·30초·PID ≤64, 비루트, 모든 capability 제거, seccomp, 키·체인 DB·다른 노드가 안 보일 것.

## 선택지

| 안 | 방법 | 장점 | 단점 |
|---|---|---|---|
| A. systemd 일시 유닛 (추천, 1단계) | Core가 `systemd-run --wait --pipe --collect` 로 호출마다 새 유닛 생성: `DynamicUser=yes`, `PrivateNetwork=yes`, `IPAddressDeny=any`, `ProtectSystem=strict`, `ProtectHome=yes`, `PrivateTmp=yes`, `PrivateDevices=yes`, `NoNewPrivileges=yes`, `CapabilityBoundingSet=`, `SystemCallFilter=@system-service`, `SystemCallArchitectures=native`, `MemoryMax=512M`, `TasksMax=64`, `RuntimeMaxSec=30`, `CPUQuota=100%`, `InaccessiblePaths=/opt/aichain/.env /opt/aichain/data /root /home` | 추가 설치 없음. 이미 브라우저 서비스에 쓴 것과 같은 방식이라 검증됨. 네트워크 공간을 systemd(root)가 만들어서 AppArmor 제약을 받지 않음 | 커널을 공유함(커널 취약점에 노출). 호출마다 수십 ms 오버헤드 |
| B. Docker + gVisor (설계 문서 기본안, 2단계) | docker, runsc 설치. `--runtime=runsc --network none --read-only --tmpfs /work --memory 512m --pids-limit 64 --cap-drop ALL --security-opt no-new-privileges` | 사용자 공간 커널(gVisor)로 커널 공격면이 크게 줄어듦 | 설치·운영 부담(Docker 데몬은 root, 이 서버의 다른 서비스와 공존), 호출당 수백 ms |
| C. bubblewrap/nsjail + 전용 사용자 | AppArmor에 bwrap 프로필 허용 필요(Ubuntu는 `bwrap-userns-restrict` 제공) | 가벼움 | AppArmor 설정 변경 필요, 설정 실수 여지 |

## 제안

1. 지금: 실행 비활성 유지. 정의·정적 검사·버전 관리만 사용(현재 상태).
2. 결정되면 **A안**을 `SystemdExecutor`로 구현하고, 탈출 시도 코드 모음(네트워크 접속, 파일 읽기 `/opt/aichain/.env`, 포크 폭탄, 메모리 폭주, tailscale IP 접속)이 전부 막히는지 테스트한 뒤 켠다. 켜는 스위치는 운영자만 바꿀 수 있는 설정으로 둔다.
3. 노드 간 도구 거래(M5)로 다른 노드의 코드를 실행하게 되는 시점에는 **B안**(gVisor)으로 올린다.

결정할 것: A안으로 실행을 켤지, 아니면 B안까지 기다릴지.

## A안 검증 결과 (2026-10-11, 서버)

`deploy/scripts/executor-escape-tests.py`: 악성 도구가 할 법한 코드 19가지를 정적 검사를 건너뛰고 실행기에 직접 넣었다. **19/19 차단**, 서버 상태 정상, 남은 유닛 없음.

| 시도 | 결과 |
|---|---|
| 인터넷·tailscale·loopback TCP 접속 | 소켓 생성 불가 (AF_INET 금지) |
| DNS 조회 | 실패 (네트워크 없음) |
| 브라우저 API·egress 프록시 Unix 소켓 | 접근 거부 |
| `/opt/aichain/.env`, 저널 DB, `/root/.ssh`, `/etc/shadow` 읽기 | 접근 거부 |
| `/etc`, `/opt` 쓰기 | 읽기 전용 |
| 환경 변수의 비밀값 | 없음 (systemd 기본 변수만) |
| 다른 프로세스 보기 | 자기 프로세스만 보임 |
| 포크 폭탄 | TasksMax=64에서 멈춤 |
| 메모리 2GB | MemoryError |
| 무한 루프 | RuntimeMaxSec에서 종료 |
| setuid(0), mount | 실패 |

호출당 오버헤드는 약 0.1초. 참고: transient 유닛에서는 `RestrictAddressFamilies=none`이 거부되어 `AF_UNIX`만 허용했다(Unix 소켓 경로는 InaccessiblePaths와 파일 권한으로 막힘).

## 허용 패키지 (2026-10-13)

문제: 실행기가 `/usr/bin/python3 -I -S`로 돌아서(`-S`는 site/dist-packages를 sys.path에 넣지 않음) 매니페스트가 허용하는 `numpy`, `pandas`, `jsonschema` 등을 쓰는 도구가 자기 테스트에서 `ModuleNotFoundError`로 실패했다.

방식: 허용 패키지만 담은 전용 디렉터리 하나를 쓴다.

- 목록: `deploy/tool-packages.txt` (버전 고정, `manifest.ALLOWED_PACKAGES`와 같아야 함 — 테스트가 확인). 의존 패키지(pytz, tzdata, attrs, mpmath, soupsieve 등)는 함께 설치된다.
- 위치: `/opt/aichain-tool-packages` (`executor.TOOL_PACKAGES_DIR`). root 소유, 파일 644·디렉터리 755. 유닛 안에서는 `ProtectSystem=strict`로 읽기 전용이다. `/opt/aichain`(InaccessiblePaths)과는 다른 경로다.
- 실행: 여전히 `python3 -I -S`. 러너가 Core가 넘긴 이 디렉터리 하나만 sys.path **끝에** 붙인다(표준 라이브러리를 가리지 못함). 시스템 Python 패키지(`/usr/lib/python3/dist-packages`, `/usr/local/lib/python3.12/dist-packages`)는 InaccessiblePaths로 막아서 sys.path를 조작해도 읽을 수 없다.
- 격리는 그대로다: 네트워크 없음, 읽기 전용 FS, `/opt/aichain`·비밀값 접근 불가. 패키지 디렉터리에 쓰기 시도도 탈출 테스트에 추가했다.
- BLAS 스레드는 1개로 고정(`OPENBLAS_NUM_THREADS=1` 등, CPU 1개·TasksMax=64).
- 메모리: numpy/pandas는 실제로 쓰는 것보다 훨씬 큰 가상 주소 공간을 예약한다. systemd 실행기에서는 cgroup `MemoryMax`(매니페스트 `memory_mb`)가 실제 메모리 상한이고, 러너의 `RLIMIT_AS`는 `memory_mb + 1024MB`로 폭주 예약만 막는다. 2GB 메모리 폭탄은 여전히 차단된다.
- 패키지가 아직 설치되지 않은 서버에서는 해당 도구 오류에 "not installed on this server yet"이 붙는다.

### 서버에서 한 번 할 일 (root, `/opt/aichain`에서)

```bash
sh deploy/scripts/install-tool-packages.sh          # uv 또는 python3 -m pip 필요 (없으면 apt install python3-pip)
.venv/bin/python deploy/scripts/executor-escape-tests.py   # 공격 21개 차단 + 허용 패키지 8개 import 확인
```

설치 스크립트는 시스템 python3(Ubuntu 24.04: 3.12)용 휠을 `/opt/aichain-tool-packages.new`에 받고, 미리 컴파일(.pyc — 샌드박스는 `__pycache__`를 쓸 수 없음)하고, `-I -S`로 import를 확인한 뒤 원래 자리로 바꿔 넣는다. `deploy/tool-packages.txt`를 바꾸면(거버넌스 `policy` 제안 후) 다시 실행한다. 서비스 재시작은 필요 없다(호출마다 새 유닛). 탈출 테스트의 패키지 항목이 FAIL이면(예: 어떤 휠의 공유 라이브러리가 실행 가능 스택을 요구해 `MemoryDenyWriteExecute=yes`와 부딪히는 경우) 격리 속성을 풀지 말고 그 패키지 버전을 바꾸거나 목록에서 빼는 쪽으로 정한다.
