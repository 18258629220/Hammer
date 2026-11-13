import os
import copy
import random
import hashlib
import math
from torch.utils.data import Dataset

import numpy as np
import torch
from PIL import Image

from src.models.point_model.pointnet2 import pc_normalize

from .utils import OAFFORD_QUESTION_LIST, OAFFORD_ANSWER_LIST
from .utils import OAFFORD_AFFORD_QUESTION_LIST, OAFFORD_AFFORD_ANSWER_LIST
from .utils import OAFFORD_AFFORD_OBJ_ANSWER_LIST
from .utils import CRAG_ROLE_ANSWER_LIST

from .conversation import get_default_conv_template
from .partfield_cache import cache_manifest, canonical_points, load_partfield_cache


AFFORD_LABEL = [
    'grasp', 'contain', 'lift', 'open', 'lay', 'sit', 'support', 'wrapgrasp', 
    'pour', 'move', 'display', 'push', 'listen', 'wear', 'press', 'cut', 'stab', 
    'carry', 'ride', 'clean', 'play', 'beat', 'speak', 'pull'
]  # 24

UNSEEN_AFFORD = [
    'Bag', 'Microphone', 'Toothbrush', 'TrashCan', 'Bicycle', 'Guitar', 
    'Glasses', 'Hat', 'Microwave', 'Door', 'Scissors','Bowl', 'Baseballbat', 
    'Mop', 'Dishwasher', 'Bed', 'Keyboard', 'Clock', 'Vase', 'Knife', 
    'Hammer', 'Refrigerator', 'Chair', 'Umbrella', 'Bucket', 'Display', 
    'Earphone', 'Motorcycle', 'StorageFurniture', 'Fork', 'Broom', 'Skateboard', 
    'Tennisracket', 'Laptop', 'Table', 'Bottle', 'Faucet', 'Kettle', 'Surfboard', 
    'Mug', 'Spoon'
]

UNSEEN_OBJ = [
    'Bag', 'Microphone', 'Toothbrush', 'TrashCan', 'Bicycle', 'Guitar', 
    'Glasses', 'Hat', 'Microwave', 'Backpack', 'Door', 'Bowl', 'Dishwasher', 
    'Bed', 'Keyboard', 'Vase', 'Knife', 'Suitcase', 'Hammer', 'Chair', 
    'Umbrella', 'Display', 'Earphone', 'StorageFurniture', 'Broom', 'Tennisracket', 
    'Table', 'Bottle', 'Faucet', 'Surfboard', 'Mug', 'Spoon'
]

SEEN = [
    'Bag', 'Microphone', 'Toothbrush', 'TrashCan', 'Bicycle', 'Guitar', 
    'Glasses', 'Hat', 'Microwave', 'Backpack', 'Door', 'Scissors', 'Bowl', 
    'Baseballbat', 'Mop',  'Dishwasher', 'Bed', 'Keyboard', 'Clock', 'Vase', 
    'Knife', 'Suitcase', 'Hammer', 'Refrigerator', 'Chair', 'Umbrella', 'Bucket', 
    'Display', 'Earphone', 'Motorcycle', 'StorageFurniture', 'Fork', 'Broom', 
    'Skateboard', 'Tennisracket', 'Laptop', 'Table', 'Bottle', 'Faucet', 
    'Kettle', 'Surfboard', 'Mug', 'Spoon'
]

ROLE_VARIANTS = ("crim", "crag3d")
ROLE_METADATA_FIELDS = {"affordance_id", "image_id", "point_canonical_indices"}


def _prepare_partfield_cache(model_variant, directory):
    if model_variant not in ("hammer", "partfield", "crim", "crag3d"):
        raise ValueError(f"Unknown dataset model variant: {model_variant}")
    if model_variant == "hammer":
        return ""
    if not directory:
        raise ValueError(f"model_variant={model_variant} requires partfield_cache_dir")
    directory = os.path.abspath(os.path.expanduser(directory))
    cache_manifest(directory)
    return directory


def _role_metadata(cached, variant, raw_xyz, affordance, image):
    if variant not in ROLE_VARIANTS:
        return cached
    if cached is None:
        raise ValueError("Role supervision requires aligned PartField caches")
    _, _, inverse = canonical_points(raw_xyz)
    image_digest = hashlib.sha256()
    image_digest.update(b"RGB\0")
    for dimension in image.size:
        image_digest.update(int(dimension).to_bytes(8, "big"))
    image_digest.update(image.tobytes())
    image_id = int.from_bytes(image_digest.digest()[:8], "big")
    cached.update(
        affordance_id=torch.tensor(AFFORD_LABEL.index(affordance.lower()), dtype=torch.long),
        image_id=torch.tensor(image_id & ((1 << 63) - 1), dtype=torch.long),
        point_canonical_indices=torch.from_numpy(inverse).long(),
    )
    return cached


def _validate_cf_batch_probability(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError("cf_batch_probability must be finite and in [0,1]")
    return float(value)


def _prepare_cf_pairs(dataset, dataset_name, index_path):
    dataset.cf_index_path = ""
    dataset.cf_pairs = []
    if not index_path:
        return
    if dataset.split != "train":
        raise ValueError("Counterfactual indexes are restricted to the training split")
    if dataset.model_variant not in ROLE_VARIANTS:
        raise ValueError("Counterfactual sampling requires a CRIM role-token variant")
    from src.models.crag.counterfactual import load_cf_index
    dataset.cf_index_path = os.path.abspath(os.path.expanduser(index_path))
    index = load_cf_index(dataset.cf_index_path, dataset.data_root, dataset_name,
                          dataset.setting, dataset.point_files)
    dataset.cf_pairs = index["pairs"]
    dataset.cf_overlap_threshold = index["overlap_threshold"]
    dataset.cf_statistics = index["statistics"]
    if not dataset.cf_pairs:
        raise ValueError("Counterfactual sampling was enabled but no legal training pairs exist")


def _sample_batch(dataset, indices):
    if not dataset.cf_pairs:
        return [dataset[index] for index in indices]
    if len(indices) < 2 or len(indices) % 2:
        raise ValueError("Counterfactual sampling requires an even local batch size of at least two")
    probability = getattr(dataset, "cf_batch_probability", 1.0)
    # Mix complete ordinary batches with complete CF batches so single-task
    # shapes remain trainable. Endpoints consume no extra routing RNG draw.
    if probability == 0 or (probability < 1 and random.random() >= probability):
        return [dataset[index] for index in indices]
    samples = []
    for _ in range(len(indices) // 2):
        pair = random.choice(dataset.cf_pairs)
        first = dataset._sample_point_idx(pair["point_a"])
        second = dataset._sample_point_idx(pair["point_b"])
        if first[5]["point_key"] != pair["point_key"] or second[5]["point_key"] != pair["point_key"]:
            raise ValueError("Counterfactual shape coordinates changed after index construction")
        if first[5]["affordance_id"].item() == second[5]["affordance_id"].item():
            raise ValueError("Counterfactual samples must use different affordances")
        samples.extend((first, second))
    return samples


def collate_fn(
    batch, tokenizer=None, processor=None, model_name="qwen_vl", role_token_ids=None
):
    """
    Custom collate function for Qwen-VL model to process images once and handle multiple conversations efficiently.

    Args:
        batch: Five-field HAMMER samples, optionally followed by a PartField cache dictionary.
        processor: Instance of Qwen2_5_VLProcessor or similar
        tokenizer: Optional tokenizer (defaults to processor.tokenizer)
        model_name: Name of the model (e.g., "qwen_vl")

    Returns:
        Dictionary with batched inputs matching the expected format.
    """
    tokenizer = tokenizer or processor.tokenizer
    model_name = model_name.lower()
    sample_lengths = {len(sample) for sample in batch}
    if sample_lengths not in ({5}, {6}):
        raise ValueError("Each batch must contain only HAMMER 5-tuples or only cached 6-tuples")
    has_partfield = sample_lengths == {6}

    image_list = []
    point_list = []
    afford_list = []
    conversation_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    feature_list, part_id_list, point_keys = [], [], []
    affordance_ids, image_ids, canonical_indices = [], [], []
    has_roles = None
    feature_shape = None

    for sample in batch:
        image, point, afford, conversation, inference = sample[:5]
        if has_partfield:
            cached = sample[5]
            required = {"partfield_features", "partfield_part_ids", "point_key"}
            if not isinstance(cached, dict) or not required.issubset(cached):
                raise ValueError("Cached samples must contain PartField features, part IDs and point_key")
            features, part_ids = cached["partfield_features"], cached["partfield_part_ids"]
            if not isinstance(features, torch.Tensor) or not isinstance(part_ids, torch.Tensor):
                raise ValueError("PartField feature and part ID fields must be tensors")
            if (point.ndim != 2 or point.shape[0] != 3 or features.ndim != 2
                    or features.shape[0] != point.shape[1] or features.shape[1] < 1
                    or part_ids.shape != (point.shape[1],) or afford.shape[0] != point.shape[1]):
                raise ValueError("Point, GT, PartField feature and part ID dimensions do not align")
            if feature_shape is not None and tuple(features.shape) != feature_shape:
                raise ValueError("PartField point count and feature dimension must agree within a batch")
            if not isinstance(cached["point_key"], str) or not cached["point_key"]:
                raise ValueError("Cached samples must have a nonempty point_key")
            feature_shape = tuple(features.shape)
            feature_list.append(features)
            part_id_list.append(part_ids)
            point_keys.append(cached["point_key"])
            role_fields = ROLE_METADATA_FIELDS.intersection(cached)
            if role_fields and role_fields != ROLE_METADATA_FIELDS:
                raise ValueError("Role batches require affordance IDs, image IDs and canonical point indices")
            sample_has_roles = bool(role_fields)
            if has_roles is not None and has_roles != sample_has_roles:
                raise ValueError("Each batch must use one consistent role-supervision format")
            has_roles = sample_has_roles
            if has_roles:
                affordance_id, image_id = cached["affordance_id"], cached["image_id"]
                canonical = cached["point_canonical_indices"]
                if (not isinstance(affordance_id, torch.Tensor) or affordance_id.dtype != torch.long
                        or affordance_id.ndim != 0 or not 0 <= affordance_id.item() < len(AFFORD_LABEL)):
                    raise ValueError("Invalid role affordance ID")
                if (not isinstance(image_id, torch.Tensor) or image_id.dtype != torch.long
                        or image_id.ndim != 0 or image_id.item() < 0):
                    raise ValueError("Invalid stable role image ID")
                if (not isinstance(canonical, torch.Tensor) or canonical.dtype != torch.long
                        or canonical.shape != (point.shape[1],)
                        or not torch.equal(canonical.sort().values, torch.arange(point.shape[1]))):
                    raise ValueError("Canonical point indices must be a permutation of the input point rows")
                affordance_ids.append(affordance_id)
                image_ids.append(image_id)
                canonical_indices.append(canonical)
        conversation = [conversation]
        image_list.append(image)
        point_list.append(torch.from_numpy(point).float())
        afford_list.append(torch.from_numpy(afford).float())
        conversation_list.extend(conversation)
        cnt += len(conversation)
        offset_list.append(cnt)
        inferences.append(inference)

    points = torch.stack(point_list, dim=0)
    affords = torch.stack(afford_list, dim=0)
    if "qwen" in model_name:
        image_inputs = processor.image_processor(image_list, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"]
        image_grid_thw = image_inputs["image_grid_thw"]

        nb_token_list = []
        for i in range(len(image_grid_thw)):
            sample_image_grid_thw = image_grid_thw[i]
            nb_tokens = sample_image_grid_thw.prod()
            nb_token_list.append(nb_tokens)

        all_modified_conversations = []
        conversation_to_sample_idx = []
        duplicated_image_grid_thw = []
        for i in range(len(offset_list) - 1):
            start = offset_list[i]
            end = offset_list[i + 1]
            sample_conversations = conversation_list[start:end]

            sample_image_grid_thw = image_grid_thw[i]
            merge_length = processor.image_processor.merge_size ** 2
            nb_tokens = nb_token_list[i]
            num_placeholders = nb_tokens // merge_length

            for conv in sample_conversations:
                if processor.image_token in conv:
                    modified_conv = conv.replace(
                        processor.image_token,
                        "<|placeholder|>" * num_placeholders,
                        1
                    )
                    modified_conv = modified_conv.replace("<|placeholder|>", processor.image_token)
                else:
                    modified_conv = conv
                all_modified_conversations.append(modified_conv)
                conversation_to_sample_idx.append(i)

            num_conversations = offset_list[i + 1] - offset_list[i]
            duplicated_image_grid_thw.extend([sample_image_grid_thw] * num_conversations)

        text_inputs = tokenizer(all_modified_conversations, return_tensors="pt", padding=True)
        input_ids = text_inputs["input_ids"]
        attention_masks = text_inputs["attention_mask"]

        targets = input_ids.clone()
        targets[targets == tokenizer.pad_token_id] = -100
        assistant_start_str = "<|im_start|>assistant\n"
        assistant_end_str = "<|im_end|>"

        assistant_start_tokens = tokenizer.encode(assistant_start_str, add_special_tokens=False)
        assistant_end_tokens = tokenizer.encode(assistant_end_str, add_special_tokens=False)

        for idx in range(targets.shape[0]):
            target = targets[idx]

            start_positions = []
            for pos in range(target.size(0) - len(assistant_start_tokens) + 1):
                if target[pos: pos + len(assistant_start_tokens)].tolist() == assistant_start_tokens:
                    start_positions.append(pos + len(assistant_start_tokens))
            if not start_positions:
                raise ValueError(f"Assistant start tokens not found in sample {idx}")

            label = torch.full_like(target, -100)
            for i, start_idx in enumerate(start_positions):
                if i < len(start_positions) - 1:
                    search_end = start_positions[i + 1] - len(assistant_start_tokens)
                else:
                    search_end = target.size(0)
                found_end = False
                for pos in range(start_idx, search_end + 1):
                    if target[pos: pos + len(assistant_end_tokens)].tolist() == assistant_end_tokens:
                        end_idx = pos + 1
                        found_end = True
                        break
                if not found_end:
                    end_idx = search_end
                label[start_idx:end_idx] = target[start_idx:end_idx]
            targets[idx] = label

        if has_roles:
            if not isinstance(role_token_ids, dict) or set(role_token_ids) != {"contact", "effect", "exclude"}:
                raise ValueError("CRIM collate requires contact/effect/exclude token IDs")
            ids = list(role_token_ids.values())
            if (any(not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0 for token_id in ids)
                    or len(set(ids)) != 3):
                raise ValueError("Role token IDs must be three distinct nonnegative integers")
            for name, token_id in role_token_ids.items():
                mask = input_ids == token_id
                if not torch.all(mask.sum(dim=-1) == 1):
                    raise ValueError(f"Each conversation must contain exactly one {name} token")
                if (targets[mask] == -100).any():
                    raise ValueError(f"The {name} token must remain in the assistant CE target")
        elif role_token_ids is not None:
            raise ValueError("Role-token collate IDs were supplied without role-supervision metadata")

        cumsum_tokens = torch.cumsum(torch.tensor(nb_token_list), dim=0)
        duplicated_pixel_values = []
        for i in range(len(image_grid_thw)):
            start = cumsum_tokens[i - 1] if i > 0 else 0
            end = cumsum_tokens[i]
            image_segment = pixel_values[start:end]
            num_conversations = offset_list[i + 1] - offset_list[i]
            repeated_segment = image_segment.repeat(num_conversations, 1)
            duplicated_pixel_values.append(repeated_segment)
            
        pixel_values_batch = torch.cat(duplicated_pixel_values, dim=0)
        duplicated_image_grid_thw = torch.stack(duplicated_image_grid_thw)
    else:
        raise NotImplementedError(f"Model {model_name} is not supported in this collate function.")
    
    result = {
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "pixel_values": pixel_values_batch,
        "image_grid_thw": duplicated_image_grid_thw,
        "points": points,
        "gt_affords": affords,
        "offset": torch.LongTensor(offset_list),
        "inference": inferences[0] if inferences else False,
    }
    if has_partfield:
        result.update(
            partfield_features=torch.stack(feature_list, dim=0),
            partfield_part_ids=torch.stack(part_id_list, dim=0),
            point_key=point_keys,
        )
    if has_roles:
        result.update(
            affordance_ids=torch.stack(affordance_ids),
            image_ids=torch.stack(image_ids),
            point_canonical_indices=torch.stack(canonical_indices),
        )
    return result


class PIADAfford(Dataset):
    def __init__(
        self,
        datasets="piad",
        data_root="./data",
        samples_per_epoch=500 * 8 * 2 * 10,
        split='train',
        setting='Seen',
        model_name="qwen_vl",
        question_type='simple', # 'simple' or 'afford' or 'afford_obj'
        model_variant='hammer',
        partfield_cache_dir='',
        cf_index_path='',
        cf_batch_probability=0.125,
    ):
        super().__init__()
        self.cf_batch_probability = _validate_cf_batch_probability(cf_batch_probability)
        self.samples_per_epoch = samples_per_epoch
        self.data_root = data_root
        self.split = split
        self.setting = setting
        self.model_name = model_name.lower()
        self.question_type = question_type
        self.model_variant = model_variant
        self.partfield_cache_dir = _prepare_partfield_cache(model_variant, partfield_cache_dir)
        
        if setting == 'Seen':
            num_dict = {item: dict() for item in SEEN}
        elif setting == 'Unseen_afford':
            num_dict = {item: dict() for item in UNSEEN_AFFORD}
        elif setting == 'Unseen_obj':
            num_dict = {item: dict() for item in UNSEEN_OBJ}
        else:
            raise ValueError(
                "Invalid setting. Choose from 'Seen', 'Unseen_afford', or 'Unseen_obj'."
            )
        
        if self.question_type == 'simple':
            self.answer_list = OAFFORD_ANSWER_LIST
            self.question_list = OAFFORD_QUESTION_LIST
        elif self.question_type == 'afford':
            self.answer_list = OAFFORD_AFFORD_ANSWER_LIST
            self.question_list = OAFFORD_AFFORD_QUESTION_LIST
        elif self.question_type == 'afford_obj':
            self.answer_list = OAFFORD_AFFORD_OBJ_ANSWER_LIST
            self.question_list = OAFFORD_AFFORD_QUESTION_LIST
        if self.model_variant in ROLE_VARIANTS:
            self.answer_list = CRAG_ROLE_ANSWER_LIST

        img_path = os.path.join(data_root, f'PIADv2/{setting}/Img_{split}.txt')
        point_path = os.path.join(data_root, f'PIADv2/{setting}/Point_{split}.txt')

        self.point_files = self.read_file(point_path)

        if split == 'train':
            self.image_files, self.num_dict = self.read_file(img_path, num_dict)
            self.obj_offset = copy.deepcopy(self.num_dict)
            start = 0
            for obj in list(num_dict.keys()):
                for aff in list(num_dict[obj].keys()):
                    split = [start, start + self.num_dict[obj][aff]]
                    self.obj_offset[obj][aff] = split
                    start += self.num_dict[obj][aff]
        else:
            self.image_files = self.read_file(img_path)
        _prepare_cf_pairs(self, "PIADv2", cf_index_path)

    def read_file(self, path, num_dict=None):
        file_list = []
        with open(path,'r') as f:
            files = f.readlines()
            base_path = os.path.join(self.data_root, 'PIADv2', self.setting)
            for file in files:
                file = file.strip('\n')
                if num_dict != None:
                    obj, aff = file.split('/')[-4], file.split('/')[-2]
                    if aff not in num_dict[obj]:
                        num_dict[obj][aff] = 1
                    else:
                        num_dict[obj][aff] += 1
                if "Data/" in file:
                    file = os.path.join(base_path, *file.split('/')[2:])
                file_list.append(file)
            f.close()
        if num_dict != None:
            return file_list, num_dict
        else:
            return file_list
        
    def generate_cnversation(self, object_class, affordance):
        question_template = random.choice(self.question_list)
        question = question_template.format(class_name=object_class.lower())
        answer = random.choice(self.answer_list).format(
            affordance=affordance.lower(), class_name=object_class.lower()
        )
        conv = get_default_conv_template(self.model_name).copy()
        if "qwen" in self.model_name:
            user_message = f"<|vision_start|><|image_pad|><|vision_end>\n{question}"
        else:
            user_message = f"<image>\n{question}"
        conv.append_message(conv.roles[0], user_message)
        conv.append_message(conv.roles[1], answer)
        conversation = conv.get_prompt()
        return conversation
        
    def __len__(self):
        if self.split == 'train':
            return self.samples_per_epoch
        else:
            return len(self.point_files)

    def __getitem__(self, idx):
        idx = random.randint(0, len(self.point_files) - 1) if self.split == 'train' else idx
        return self._sample_point_idx(idx)

    def __getitems__(self, indices):
        return _sample_batch(self, indices)

    def _sample_point_idx(self, idx):
        point_file = self.point_files[idx]
        point, afford = self.extract_point_file(point_file)
        raw_point = point
        point = pc_normalize(point)
        cached = load_partfield_cache(self.partfield_cache_dir, raw_point, point) if self.partfield_cache_dir else None
        point = point.transpose()

        obj_name, afford_name = point_file.split('/')[-4], point_file.split('/')[-2]

        if self.split == 'train':
            offset = self.obj_offset[obj_name][afford_name]
            sample_idx = random.randint(offset[0], offset[1] - 1)
            img_file = self.image_files[sample_idx]
        else:
            img_file = self.image_files[idx]

        img_afford = img_file.split('/')[-2]
        assert img_afford == afford_name, \
            f"Image affordance {img_afford} does not match point affordance {afford_name}."
        image_vlm = Image.open(img_file).convert('RGB')
        cached = _role_metadata(cached, self.model_variant, raw_point, afford_name, image_vlm)
        conversation = self.generate_cnversation(obj_name, afford_name)
        inference = False if self.split == 'train' else True

        sample = image_vlm, point, afford, conversation, inference
        return sample + (cached,) if cached is not None else sample

    def extract_point_file(self, path):
        lines = np.load(path)
        data_array = np.array(lines)
        coords = data_array[:, 0:3]
        afford = data_array[:, 3:]
        return coords, afford

    def get_affordance_index(self, str):
        cut_str = str.split('/')
        affordance = cut_str[-2]
        index = AFFORD_LABEL.index(affordance)
        return index


class PIADv1Afford(Dataset):
    def __init__(
        self,
        datasets="piad",
        data_root="./data",
        samples_per_epoch=500 * 8 * 2 * 10,
        split='train',
        setting='Seen',
        model_name="qwen_vl",
        question_type='simple', # 'simple' or 'afford' or 'afford_obj'
        model_variant='hammer',
        partfield_cache_dir='',
        cf_index_path='',
        cf_batch_probability=0.125,
    ):
        super().__init__()
        self.cf_batch_probability = _validate_cf_batch_probability(cf_batch_probability)
        self.samples_per_epoch = samples_per_epoch
        self.data_root = data_root
        self.split = split
        self.setting = setting
        self.model_name = model_name.lower()
        self.question_type = question_type
        self.model_variant = model_variant
        self.partfield_cache_dir = _prepare_partfield_cache(model_variant, partfield_cache_dir)

        self.afford_label = [
            'grasp', 'contain', 'lift', 'open', 'lay', 'sit', 'support', 'wrapgrasp', 
            'pour', 'move', 'display', 'push', 'listen', 'wear', 'press', 'cut', 'stab'
        ]
        
        if setting == 'Seen':
            seen_obj = [
                'Earphone', 'Bag', 'Chair', 'Refrigerator', 'Knife', 'Dishwasher',
                'Keyboard', 'Scissors', 'Table', 'StorageFurniture', 'Bottle', 'Bowl',
                'Microwave', 'Display', 'TrashCan', 'Hat', 'Clock', 'Door', 'Mug',
                'Faucet', 'Vase', 'Laptop', 'Bed'
            ]
            num_dict = {item: dict() for item in seen_obj}
        elif setting == 'Unseen':
            unseen_obj = [
                'Knife', 'Refrigerator', 'Earphone', 'Bag', 'Keyboard', 'Chair',
                'Hat', 'Door', 'TrashCan', 'Table', 'Faucet', 'StorageFurniture',
                'Bottle', 'Bowl', 'Display', 'Mug', 'Clock'
            ]
            num_dict = {item: dict() for item in unseen_obj}
        else:
            raise ValueError(
                "Invalid setting. Choose from 'Seen', or 'Unseen'."
            )
        
        if self.question_type == 'simple':
            self.answer_list = OAFFORD_ANSWER_LIST
            self.question_list = OAFFORD_QUESTION_LIST
        elif self.question_type == 'afford':
            self.answer_list = OAFFORD_AFFORD_ANSWER_LIST
            self.question_list = OAFFORD_AFFORD_QUESTION_LIST
        elif self.question_type == 'afford_obj':
            self.answer_list = OAFFORD_AFFORD_OBJ_ANSWER_LIST
            self.question_list = OAFFORD_AFFORD_QUESTION_LIST
        if self.model_variant in ROLE_VARIANTS:
            self.answer_list = CRAG_ROLE_ANSWER_LIST

        img_path = os.path.join(data_root, f'PIADv1/{setting}/Img_{split}.txt')

        if split == 'train':
            point_path = os.path.join(data_root, f'PIADv1/{setting}/Point_Extracted_train.txt')
            self.point_files = self.read_file(point_path)
            self.image_files, self.num_dict = self.read_file(img_path, num_dict)
            self.obj_offset = copy.deepcopy(self.num_dict)
            start = 0
            for obj in list(num_dict.keys()):
                for aff in list(num_dict[obj].keys()):
                    split = [start, start + self.num_dict[obj][aff]]
                    self.obj_offset[obj][aff] = split
                    start += self.num_dict[obj][aff]
        else:
            point_path = os.path.join(data_root, f'PIADv1/{setting}/Point_test.txt')
            self.point_files = self.read_file(point_path)
            self.image_files = self.read_file(img_path)
        _prepare_cf_pairs(self, "PIADv1", cf_index_path)

    def read_file(self, path, num_dict=None):
        file_list = []
        with open(path,'r') as f:
            files = f.readlines()
            base_path = os.path.join(self.data_root, 'PIADv1', self.setting)
            for file in files:
                file = file.strip('\n')
                if num_dict != None:
                    obj, aff = file.split('/')[-3], file.split('/')[-2]
                    if aff not in num_dict[obj]:
                        num_dict[obj][aff] = 1
                    else:
                        num_dict[obj][aff] += 1
                if "Data/" in file:
                    file = os.path.join(base_path, *file.split('/')[2:])
                file_list.append(file)
            f.close()
        if num_dict != None:
            return file_list, num_dict
        else:
            return file_list
        
    def generate_cnversation(self, object_class, affordance):
        question_template = random.choice(self.question_list)
        question = question_template.format(class_name=object_class.lower())
        answer = random.choice(self.answer_list).format(
            affordance=affordance.lower(), class_name=object_class.lower()
        )
        conv = get_default_conv_template(self.model_name).copy()
        if "qwen" in self.model_name:
            user_message = f"<|vision_start|><|image_pad|><|vision_end>\n{question}"
        else:
            user_message = f"<image>\n{question}"
        conv.append_message(conv.roles[0], user_message)
        conv.append_message(conv.roles[1], answer)
        conversation = conv.get_prompt()
        return conversation
        
    def __len__(self):
        if self.split == 'train':
            return self.samples_per_epoch
        else:
            return len(self.point_files)

    def __getitem__(self, idx):
        idx = random.randint(0, len(self.point_files) - 1) if self.split == 'train' else idx
        return self._sample_point_idx(idx)

    def __getitems__(self, indices):
        return _sample_batch(self, indices)

    def _sample_point_idx(self, idx):
        point_file = self.point_files[idx]
        point, afford = self.extract_point_file(point_file)
        raw_point = point
        point = pc_normalize(point)
        cached = load_partfield_cache(self.partfield_cache_dir, raw_point, point) if self.partfield_cache_dir else None
        point = point.transpose()

        if self.split == 'train':
            obj_name, afford_name = point_file.split('/')[-3], point_file.split('/')[-2]
            offset = self.obj_offset[obj_name][afford_name]
            sample_idx = random.randint(offset[0], offset[1] - 1)
            img_file = self.image_files[sample_idx]
            img_afford = img_file.split('/')[-2]
            assert img_afford == afford_name, \
                f"Image affordance {img_afford} does not match point affordance {afford_name}."
        else:
            img_file = self.image_files[idx]
            _, afford = self.get_affordance_index(img_file, afford)
            obj_name, afford_name = img_file.split('/')[-3], img_file.split('/')[-2]
            
        image_vlm = Image.open(img_file).convert('RGB')
        cached = _role_metadata(cached, self.model_variant, raw_point, afford_name, image_vlm)
        conversation = self.generate_cnversation(obj_name, afford_name)
        inference = False if self.split == 'train' else True

        sample = image_vlm, point, afford, conversation, inference
        return sample + (cached,) if cached is not None else sample

    def extract_point_file(self, path):
        if self.split == 'train':
            lines = np.load(path)
            data_array = np.array(lines)
            coords = data_array[:, 0:3]
            afford = data_array[:, 3:]
        elif self.split == 'test':
            with open(path, 'r') as f:
                coordinates = []
                lines = f.readlines()
                for line in lines:
                    line = line.strip('\n')
                    line = line.strip(' ')
                    data = line.split(' ')
                    coordinate = [float(x) for x in data[2:]]
                    coordinates.append(coordinate)
                f.close()
            data_array = np.array(coordinates)
            coords = data_array[:, 0:3]
            afford = data_array[:, 3:]
        return coords, afford

    def get_affordance_index(self, str, afford=None):
        cut_str = str.split('/')
        affordance = cut_str[-2]
        index = self.afford_label.index(affordance)
        if afford is not None:
            afford = afford[:, index:index + 1]
        return index, afford
    
