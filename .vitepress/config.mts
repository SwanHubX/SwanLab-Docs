import { defineConfig } from 'vitepress'
import { zh } from './zh'
import { en } from './en'

export default defineConfig({
  rewrites: {
    'zh/:rest*': ':rest*'
  },

  themeConfig:{
    search: {
      provider: 'algolia',
      options: {
        appId: 'T5KXUIYB4F',
        apiKey: '43abc0172fb6a8dc6e7794a59861a573',
        indexName: 'swanlab',
        locales: {
          root: {
            placeholder: '搜索文档',
            translations: {
              button: {
                buttonText: '搜索文档',
                buttonAriaLabel: '搜索文档',
              },
              modal: {
                searchBox: {
                  resetButtonTitle: '清除查询条件',
                  resetButtonAriaLabel: '清除查询条件',
                  cancelButtonText: '取消',
                  cancelButtonAriaLabel: '取消',
                },
                startScreen: {
                  recentSearchesTitle: '搜索历史',
                  noRecentSearchesText: '没有搜索历史',
                  saveRecentSearchButtonTitle: '保存至搜索历史',
                  removeRecentSearchButtonTitle: '从搜索历史中移除',
                  favoriteSearchesTitle: '收藏',
                  removeFavoriteSearchButtonTitle: '从收藏中移除',
                },
                errorScreen: {
                  titleText: '无法获取结果',
                  helpText: '你可能需要检查你的网络连接',
                },
                footer: {
                  selectText: '选择',
                  navigateText: '切换',
                  closeText: '关闭',
                  searchByText: '搜索提供者',
                },
                noResultsScreen: {
                  noResultsText: '无法找到相关结果',
                  suggestedQueryText: '你可以尝试查询',
                  reportMissingResultsText: '你认为该查询应该有结果？',
                  reportMissingResultsLinkText: '点击反馈',
                },
              },
            },
          },
          en: {
            placeholder: 'Search Documentation',
            translations: {
              button: {
                buttonText: 'Search Documentation',
                buttonAriaLabel: 'Search Documentation',
              },
              modal: {
                searchBox: {
                  resetButtonTitle: 'Clear query',
                  resetButtonAriaLabel: 'Clear query',
                  cancelButtonText: 'Cancel',
                  cancelButtonAriaLabel: 'Cancel',
                },
                startScreen: {
                  recentSearchesTitle: 'Recent',
                  noRecentSearchesText: 'No recent searches',
                  saveRecentSearchButtonTitle: 'Save to recent searches',
                  removeRecentSearchButtonTitle: 'Remove from recent searches',
                  favoriteSearchesTitle: 'Favorite',
                  removeFavoriteSearchButtonTitle: 'Remove from favorites',
                },
                errorScreen: {
                  titleText: 'Unable to fetch results',
                  helpText: 'You might want to check your network connection',
                },
                footer: {
                  selectText: 'Select',
                  navigateText: 'Navigate',
                  closeText: 'Close',
                  searchByText: 'Search by',
                },
                noResultsScreen: {
                  noResultsText: 'No results for',
                  suggestedQueryText: 'Try searching for',
                  reportMissingResultsText:
                    'Believe this query should return results?',
                  reportMissingResultsLinkText: 'Let us know',
                },
              },
            },
          },
        },
      },
    }
  },

  markdown: {
    image: {
      lazyLoading: true
    }
  },

  locales: {
    root: { label: '简体中文', ...zh },
    en: { label: 'English', ...en },
  }
})
