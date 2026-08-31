## 2024-05-18 - Avoid repeated disk I/O for manifest loading
**학습 내용:** `loadInstallManifests` 함수가 여러 곳에서 불필요하게 파일 시스템 I/O와 `fs.readdirSync`(`addSyntheticSkillComponents` 내부의 `listSkillDirectoryIds`)를 여러 번 실행하고 있습니다. 이는 성능 저하의 원인이 됩니다. CLI 작업 중 여러 함수(`resolveInstallPlan`, `listInstallComponents` 등)가 같은 프로세스 내에서 `loadInstallManifests`를 반복 호출하기 때문입니다.

**적용 계획:** 메모리에 `cachedManifests` 변수를 두고, `repoRoot`를 기준으로 캐싱하여 불필요한 파일 I/O 및 재구성을 방지합니다.
