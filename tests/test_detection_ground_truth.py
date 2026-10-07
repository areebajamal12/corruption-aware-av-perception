from av_perception.detection.ground_truth import map_nuscenes_category


def test_requested_nuscenes_class_mapping() -> None:
    assert map_nuscenes_category("vehicle.car") == "vehicle"
    assert map_nuscenes_category("vehicle.truck") == "vehicle"
    assert map_nuscenes_category("vehicle.bus.rigid") == "vehicle"
    assert map_nuscenes_category("human.pedestrian.adult") == "pedestrian"


def test_unmapped_categories_are_excluded() -> None:
    assert map_nuscenes_category("vehicle.bicycle") is None
    assert map_nuscenes_category("vehicle.trailer") is None
    assert map_nuscenes_category("movable_object.barrier") is None
