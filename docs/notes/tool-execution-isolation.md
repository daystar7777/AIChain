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
