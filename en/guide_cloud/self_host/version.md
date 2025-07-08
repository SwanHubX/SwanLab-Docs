# Version Table

| Version | Release Date | Major Updates | Compatible Python Package Versions |
| --- | --- | --- | --- |
| v1.2 | 25-05-30 | Cloud version updated synchronously with release date | <=0.6.4 |
| v1.1 | 25-04-27 | Added offline License verification feature; <br> Cloud version updated synchronously with release date | <=0.6.4 |
| v1.0 | 25-03-12 | Initial version | <=0.5.5 |

[Upgrade Version](/en/guide_cloud/self_host/docker-deploy.html#upgrade-version)

## v1.2 (2025.5.30)

- **Feature**: Added Line Chart creation and editing feature; <br> Added Tag support for experiments; <br> Added Log Scale support for Line Charts; <br> Added Group Dragging support for Line Charts; <br> Added swanlab.OpenApi interface
- **Feature**: Added default space and default visibility configuration, used to specify the default project creation in the corresponding organization
- **Optimize**: Optimized the problem of partial data loss due to a large number of metric uploads
- **Optimize**: Significantly optimized the performance of metric uploads
- **BugFix**: Fixed the problem that experiments could not be automatically closed
