# swanlab.Molecule

[源代码](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/modules/object3d/molecule.py)

| 参数        | 描述       |
|-----------|------------------------------------------------------------------------------------------------|
| pdb_data | (str) 接收的PDB数据（字符串形式）                                 |      
| caption   | (str) 分子对象的标签。用于在实验看板中展示分子对象时进行标记。                |        

## 简介

对各种类型的生物化学分子做转换，以被`swanlab.log()`记录。

![molecule gif](/assets/molecule.gif)

## 从RDKit Mol对象创建

```python
from rdkit import Chem
import swanlab

mol = Chem.MolFromSmiles("CCO")
molecule = swanlab.Molecule.from_mol(mol, caption="Ethanol")

swanlab.init(project="molecule_demo")
swanlab.log({"molecule": molecule})
```

## 从PDB文件创建

```python
import swanlab

molecule = swanlab.Molecule.from_pdb("path/to/your/pdb/file.pdb")

swanlab.init(project="molecule_demo")
swanlab.log({"molecule": molecule})
```

## 从SDF文件创建

```python
import swanlab

molecule = swanlab.Molecule.from_sdf("path/to/your/sdf/file.sdf")

swanlab.init(project="molecule_demo")
swanlab.log({"molecule": molecule})
```

## 从SMILES字符串创建

```python
import swanlab

molecule = swanlab.Molecule.from_smiles("CCO")

swanlab.init(project="molecule_demo")
swanlab.log({"molecule": molecule})
```

## 从MOL文件创建

```python
import swanlab

molecule = swanlab.Molecule.from_mol("path/to/your/mol/file.mol")

swanlab.init(project="molecule_demo")
swanlab.log({"molecule": molecule})
```