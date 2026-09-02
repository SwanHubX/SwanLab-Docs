APP_VERSION_FILE := APP_VERSION
APP_VERSION := $(shell cat $(APP_VERSION_FILE) | tr -d '[:space:]')

# Docs files containing hardcoded SwanLab app version
DOC_FILES_ZH := zh/self_host/docker/offline-deployment.md zh/self_host/docker/deploy.md zh/self_host/kubernetes/deploy.md
DOC_FILES_EN := en/self_host/docker/offline-deployment.md en/self_host/docker/deploy.md en/self_host/kubernetes/deploy.md
DOC_FILES := $(DOC_FILES_ZH) $(DOC_FILES_EN)

.PHONY: bump-version verify-version

## Show current APP_VERSION
show-version:
	@echo "Current APP_VERSION: $(APP_VERSION)"

## Bump APP_VERSION and update all docs
## Usage: make bump-version V=v2.9.0（V=2.9.0 也可，会自动补 v 前缀；应用镜像 tag 始终带 v）
bump-version:
ifndef V
	$(error Usage: make bump-version V=v2.9.0)
endif
	@case "$(V)" in v*) NV="$(V)" ;; *) NV="v$(V)" ;; esac; \
	echo "$$NV" > $(APP_VERSION_FILE); \
	echo "✅ Updated $(APP_VERSION_FILE) to $$NV"; \
	$(MAKE) _replace-versions NEW_V=$$NV

## Replace version strings in docs
## 注：sed 中 v 前缀可缺省（v\{0,1\}），用于修复历史上漏写 v 的镜像 tag
_replace-versions:
	@for f in $(DOC_FILES); do \
		if [ -f "$$f" ]; then \
			sed -i '' 's|swanlab-server:v\{0,1\}[0-9][0-9.]*|swanlab-server:$(NEW_V)|g' "$$f"; \
			sed -i '' 's|swanlab-house:v\{0,1\}[0-9][0-9.]*|swanlab-house:$(NEW_V)|g' "$$f"; \
			sed -i '' 's|swanlab-cloud:v\{0,1\}[0-9][0-9.]*|swanlab-cloud:$(NEW_V)|g' "$$f"; \
			sed -i '' 's|swanlab-next:v\{0,1\}[0-9][0-9.]*|swanlab-next:$(NEW_V)|g' "$$f"; \
			sed -i '' 's|Self-Hosted Docker v\{0,1\}[0-9][0-9.]*|Self-Hosted Docker $(NEW_V)|g' "$$f"; \
			sed -i '' 's|<VersionBadge version="v\{0,1\}[0-9][0-9.]*" />|<VersionBadge version="$(NEW_V)" />|g' "$$f"; \
			echo "  ✅ Updated $$f"; \
		fi; \
	done

## Verify docs version matches APP_VERSION (for CI)
verify-version:
	@echo "🔍 Verifying docs match APP_VERSION=$(APP_VERSION)..."
	@FAILED=0; \
	for f in $(DOC_FILES); do \
		if [ -f "$$f" ]; then \
			MISMATCH=$$(grep -nE 'swanlab-(server|house|cloud|next):v?[0-9]' "$$f" | grep -v ":$(APP_VERSION)" || true); \
			if [ -n "$$MISMATCH" ]; then \
				echo "❌ Version mismatch in $$f:"; \
				echo "$$MISMATCH"; \
				FAILED=1; \
			fi; \
			BANNER_MISMATCH=$$(grep -nE 'Self-Hosted Docker v?[0-9]' "$$f" | grep -v " $(APP_VERSION)" || true); \
			if [ -n "$$BANNER_MISMATCH" ]; then \
				echo "❌ Banner version mismatch in $$f:"; \
				echo "$$BANNER_MISMATCH"; \
				FAILED=1; \
			fi; \
			BADGE_MISMATCH=$$(grep -nE '<VersionBadge version="v?[0-9]' "$$f" | grep -v "\"$(APP_VERSION)\"" || true); \
			if [ -n "$$BADGE_MISMATCH" ]; then \
				echo "❌ Badge version mismatch in $$f:"; \
				echo "$$BADGE_MISMATCH"; \
				FAILED=1; \
			fi; \
		fi; \
	done; \
	if [ "$$FAILED" = "1" ]; then \
		echo ""; \
		echo "Run 'make bump-version V=$(APP_VERSION)' to fix."; \
		exit 1; \
	fi
	@echo "✅ All docs match $(APP_VERSION)"
