# 关于swanlab域名迁移导致的无法访问解决方法


亲爱的SwanLab用户：

您好！

感谢您一直以来对SwanLab的支持和关注。我们计划于**2024年12月15日至12月30日**对[https://swanlab.cn](https://swanlab.cn)进行域名备案。在此期间，可能会出现[https://swanlab.cn](https://swanlab.cn)卡顿或者无法访问的情况。在这段时间建临时使用 [https://swanlab.115.zone](https://swanlab.115.zone) 来快速访问网站，**预计24年12月底迁移完成后域名访问恢复正常，同时此临时域名将保留3个月**。感谢您的理解和支持！

我们团队尽全力保障服务的稳定性和可持续性，但无奈重新备案域名管局要求必须保障现有域名无法访问。本次迁移python包不受影响，这以为着您正在运行的实验以及您的代码无需进行任何更改。当然我们还是推荐您使用`pip install -U swanlab`命令来更新最新的python包，以体验最新的功能和保障训练的稳定性。此外本教程网站同样不受影响，您可以随时访问此页面查看我们的临时域名，仅需记住[https://docs.swanlab.cn](https://docs.swanlab.cn)

我们将尽全力缩短维护时间，并确保网站尽快恢复正常运行。**如果您在此期间有任何问题或需要帮助，请随时通过[contact@swanlab.cn]与我们联系**，或者通过下方二维码加入社群或者飞书群联系我们，我们将第一时间为您解决遇到的问题！我们诚挚的邀请您关注swanlab公众号（见下图），我们将在公众号中发布最新的论文、训练教程等。

祝训练loss永不NAN！

陈少宏-SwanLab团队成员

## 社群

- **微信交流群**：交流使用SwanLab的问题、分享最新的AI技术。

<div align="center">
<img src="/assets/wechat-QR-Code.png" width=300>

</div>

- **飞书群**：我们的日常工作交流在飞书上，飞书群的回复会更及时。用飞书App扫描下方二维码即可：

<div align="center">
<img src="/assets/feishu-QR-Code.png" width=300>
</div>

## 社交媒体

- **微信公众号**:

<div align="center">
<img src="/assets/wechat_public_account.jpg" width=300>

</div>

## 功能影响跟踪表

| **功能名称**     | **可能受到的影响**                | **解决方案**                      |
|-------------------|-----------------------------------|-------------------------------|
| python包实验记录功能         | 不受影响             | 建议`pip install -U swanlab`更新至最新版体验更新的          |
| python包开启实验显示链接         | 出现的链接可能出现延时增加或者无法访问             | 请使用临时域名[https://swanlab.115.zone](https://swanlab.115.zone)查看对应的实验，预计将于24年12月30日完成迁移          |
| python包离线使用         | 不受影响             | 不受影响          |
| 网站访问         | 可能会出现延迟增加或者无法访问的问题                   | 请使用临时域名[https://swanlab.115.zone](https://swanlab.115.zone)临时访问，预计将于24年12月30日完成迁移        |
| 网站重新登录         | 使用[https://swanlab.115.zone](https://swanlab.115.zone)域名访问需要重新登陆网站                 | 无      |
| 教程文档         | 不受影响             | 可通过[https://docs.swanlab.cn](https://docs.swanlab.cn)直接访问          |
