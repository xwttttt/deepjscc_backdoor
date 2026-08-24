import ast
import os
import random
import shutil
import sys

import numpy as np
import torch

# ==================================================
#
# ==================================================
_UINT8_MAX_F = float(torch.iinfo(torch.uint8).max)
_EPS = 1e-6


def numpy2tensor(image: np.ndarray, dtype: torch.dtype = None, range_min: int = -1) -> torch.Tensor:
    """Converts uint8 image array to torch tensor.

    Args:
        image_array:
            - single: HxWx3 (ndim=3)
            - batch: BxHxWx3 (ndim=4)
        dtype: target torch dtype
        range_min: -1 → [-1,1]; else keep [0,1]
    Returns:
        single: 3xHxW; batch: Bx3xHxW
    """
    ndim = image.ndim
    if ndim == 4:
        # batch B,H,W,3 → B,3,H,W
        indices = list(range(1, ndim))[-1:] + list(range(1, ndim))[:-1]
        image = image.transpose((0,) + tuple(indices)) / _UINT8_MAX_F
    elif ndim == 3:
        # single H,W,3 → 3,H,W
        image = image.transpose(2, 0, 1) / _UINT8_MAX_F
    else:
        raise ValueError(f"Expect ndim=3(HWC) or 4(BHWC), got ndim={ndim}")

    if range_min == -1:
        image = 2.0 * image - 1.0
    return torch.from_numpy(image).to(getattr(torch, dtype))


def tensor2numpy(tensor: torch.Tensor, range_min: int = -1) -> np.ndarray:
    if range_min == -1:
        tensor = (tensor.float() + 1.0) / 2.0
    ndim = tensor.ndim
    image = tensor.clamp(0, 1).detach().cpu().numpy()
    if ndim == 4:
        # batch B,3,H,W → B,H,W,3
        image = image.transpose((0,) + tuple(range(2, ndim)) + (1,))
    elif ndim == 3:
        # single 3,H,W → H,W,3
        image = image.transpose(1, 2, 0)
    else:
        raise ValueError(f"Expect ndim=3(HWC) or 4(BHWC), got ndim={ndim}")

    return (image * _UINT8_MAX_F + 0.5).astype(np.uint8)


# ==================================================
#
# ==================================================


class Archiver:
    def __init__(self, project_root, target_dir=None):
        self.project_root = os.path.abspath(project_root)
        self.target_dir = os.path.abspath(target_dir or "./archive")
        self.main_script_path = None

        os.makedirs(self.target_dir, exist_ok=True)

    def archive_main_script(self):
        """保存主脚本"""
        main_script_path = os.path.abspath(sys.argv[0])
        target_main_script_path = os.path.join(
            self.target_dir,
            os.path.basename(main_script_path),
        )
        shutil.copy2(main_script_path, target_main_script_path)
        self.main_script_path = main_script_path

    def archive_imported_modules(self):
        """递归解析 import 语句，保存项目内自定义引用的 .py 文件"""
        if not self.main_script_path:
            raise ValueError("main_script_path 未设置，请先执行 archive_main_script()")

        visited = set()  # 已解析过的源码文件
        imported_files = set()  # 所有需要复制的文件

        def resolve_local_module(module_name, current_file=None):
            """
            将 import 的模块名解析为项目内 .py 文件路径
            只处理 project_root 下的自定义模块
            """
            if not module_name:
                return None

            # 优先按 project_root 绝对定位
            module_path = module_name.replace(".", os.sep) + ".py"
            full_path = os.path.join(self.project_root, module_path)
            if os.path.isfile(full_path):
                return os.path.abspath(full_path)

            # 兼容当前文件同目录下的相对组织形式
            if current_file:
                current_dir = os.path.dirname(os.path.abspath(current_file))
                candidate = os.path.join(current_dir, module_name.replace(".", os.sep) + ".py")
                if os.path.isfile(candidate):
                    candidate = os.path.abspath(candidate)
                    if os.path.commonpath([candidate, self.project_root]) == self.project_root:
                        return candidate

            return None

        def parse_file(file_path):
            """递归解析某个文件里的本地 import"""
            file_path = os.path.abspath(file_path)

            if file_path in visited:
                return
            visited.add(file_path)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read(), filename=file_path)
            except Exception as e:
                print(f"✗ 解析失败: {file_path} ({e})")
                return

            for node in ast.walk(tree):
                # 处理: import xxx
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        full_path = resolve_local_module(alias.name, current_file=file_path)
                        if full_path and full_path not in imported_files:
                            imported_files.add(full_path)
                            parse_file(full_path)

                # 处理: from xxx import yyy
                elif isinstance(node, ast.ImportFrom):
                    # 先处理绝对导入: from package.module import xxx
                    if node.module:
                        full_path = resolve_local_module(node.module, current_file=file_path)
                        if full_path and full_path not in imported_files:
                            imported_files.add(full_path)
                            parse_file(full_path)

                    # 再处理相对导入: from . import xxx / from ..subpackage import yyy
                    if node.level and file_path:
                        base_dir = os.path.dirname(file_path)

                        # 根据相对层级回退目录
                        rel_base = base_dir
                        for _ in range(node.level - 1):
                            rel_base = os.path.dirname(rel_base)

                        if node.module:
                            module_base_dir = os.path.join(rel_base, node.module.replace(".", os.sep))
                        else:
                            module_base_dir = rel_base

                        for alias in node.names:
                            # from . import a  => 当前目录/a.py
                            # from .subpackage import b => 当前目录/subpackage/b.py
                            candidate = os.path.join(module_base_dir, alias.name.replace(".", os.sep) + ".py")
                            if os.path.isfile(candidate):
                                candidate = os.path.abspath(candidate)
                                if os.path.commonpath([candidate, self.project_root]) == self.project_root:
                                    if candidate not in imported_files:
                                        imported_files.add(candidate)
                                        parse_file(candidate)

        # 从主脚本开始递归
        parse_file(self.main_script_path)

        # 复制所有找到的自定义模块
        for src_path in imported_files:
            rel_path = os.path.relpath(src_path, self.project_root)
            dst_path = os.path.join(self.target_dir, rel_path)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)
            print(f"✓ 已保存模块: {src_path}")

    def archive_extra_files(self, file_list):
        for file_path in file_list:
            if os.path.exists(file_path):
                dst = os.path.join(self.target_dir, os.path.basename(file_path))
                shutil.copy2(file_path, dst)
                print(f"✓ 附加文件已保存: {file_path}")
            else:
                print(f"✗ 文件不存在: {file_path}")


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def norm_power(iq):
    power = torch.mean(torch.abs(iq) ** 2, dim=-1, keepdim=True)
    return iq * torch.rsqrt(power.clamp_min(_EPS))
