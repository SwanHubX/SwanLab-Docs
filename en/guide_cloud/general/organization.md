# Using SwanLab as a Team

:::warning 

Organization creation is now fully open, with a maximum of 15 people per organization.

:::

## Create an organization

At the bottom left of the homepage, click the "Create a new organization" button and fill in the organization name to complete the organization creation.


<div align="center">
<img src="/assets/organization-create.jpg" width="400">
</div>

## Upload experiments to the organization space

By default, your project will be uploaded to your personal space. 
To upload to the organization space, set the `workspace` parameter of `swanlab.init` to the organization's name (not the organization's nickname).

```python
import swanlab

swanlab.init(
    workspace="[organization name]"
)
```

If multiple people in an organization want to collaborate on a project, simply set the `project` parameter of `swanlab.init` to the same one.
