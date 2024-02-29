import carla
import numpy as np
from skimage.measure import label, regionprops

class Helper:
    def __init__(self):
        self.gt_labels = None

    def get_image_point(self,loc, K, w2c):

        point = np.array([loc.x, loc.y, loc.z, 1])

        point_camera = np.dot(w2c, point)

        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

        point_img = np.dot(K, point_camera)

        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]

        return point_img[0:2]

    def build_projection_matrix(self,w, h, fov):
        focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = w / 2.0
        K[1, 2] = h / 2.0
        return K

    def get_vehicles_mask(self,detected_vehicle_mask,segmentation,detected_id):
        vehicles_ids = [13,14,15,16,18,19]
        for vid in vehicles_ids:
            if vid == detected_id:
                continue
            vid_mask = (segmentation == vid)
            detected_vehicle_mask = np.logical_or(detected_vehicle_mask,vid_mask)
        return detected_vehicle_mask


    def is_valid_bbox(self,bbox,segmentation,type,type_pixels_thresh,type_pixels_zero_thresh):
        segmentation = segmentation[:,:,0]
        type_map = {"person":12,"vehicle":14,"truck":15,"bus":16,"traffic_light":7,"traffic_signs":8,"motorcycle":18,"bicycle":19,"rider":13}

        type_max_id = type_map[type]
        type_mask = (segmentation == type_max_id)
        if type_map[type] in [13,14,15,16,18,19]:
            type_mask = self.get_vehicles_mask(type_mask,segmentation,type_map[type])
        xmin, ymin, xmax, ymax = bbox
        bottom_right = (int(xmax), int(ymax))
        top_left = (int(xmin), int(ymin))
        roi = type_mask.astype(np.uint8)[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        count_true_pixels = np.sum(roi == 1)
        count_false_pixels = np.sum(roi == 0)
        if count_true_pixels > type_pixels_thresh[type] and count_false_pixels > type_pixels_zero_thresh[type]:
            return True
        else:
            return False

    def is_bbox_overlaping(self,bbox,bbox_list):
        for bb in bbox_list:
            outer_x1, outer_y1, outer_x2, outer_y2 = bbox
            inner_x1, inner_y1, inner_x2, inner_y2 = bb
            is_inside = (outer_x1 <= inner_x1) and (outer_y1 <= inner_y1) and (outer_x2 >= inner_x2) and (outer_y2 >= inner_y2)

            if is_inside:
                return True
        return False

    def bbox_from_mask(self,type,carla_config):
        type_pixels_thresh = dict(carla_config['dataset_settings']['object_class_numpixel_threshold'])
        type_map = {"person": 12, "vehicle": 14, "truck": 15, "bus": 16, "traffic_light": 7, "traffic_signs": 8,"motorcycle": 18, "bicycle": 19,"rider":13}
        type_mask = (self.gt_labels[:,:,0] == type_map[type])

        lbl_0 = label(type_mask)
        props = regionprops(lbl_0)
        detected_bboxes = []
        detected_bboxes_names = []
        for prop in props:
            x_min = prop.bbox[1]
            y_min = prop.bbox[0]
            x_max = prop.bbox[3]
            y_max = prop.bbox[2]
            bottom_right = (int(x_max), int(y_max))
            top_left = (int(x_min), int(y_min))
            roi = type_mask.astype(np.uint8)[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            count_true_pixels = np.sum(roi == 1)
            if count_true_pixels > type_pixels_thresh[type] and (roi.shape[0] * roi.shape[1]) > 600:
                detected_bboxes.append([int(x_min), int(y_min), int(x_max), int(y_max)])
                detected_bboxes_names.append('rider')
        return detected_bboxes,detected_bboxes_names


    #code from https://github.com/carla-simulator/carla/issues/3801
    def get_bounding_box(self, actor, min_extent=0.5):
        """
        Some actors like motorbikes have a zero width bounding box. This is a fix to this issue.
        """
        if not hasattr(actor, "bounding_box"):
            return carla.BoundingBox(carla.Location(0, 0, min_extent),
                                     carla.Vector3D(x=min_extent, y=min_extent, z=min_extent))

        bbox = actor.bounding_box

        buggy_bbox = (bbox.extent.x * bbox.extent.y * bbox.extent.z == 0)
        if buggy_bbox:
            bbox.location = carla.Location(0, 0, max(bbox.extent.z, min_extent))

        if bbox.extent.x < min_extent:
            bbox.extent.x = min_extent
        if bbox.extent.y < min_extent:
            bbox.extent.y = min_extent
        if bbox.extent.z < min_extent:
            bbox.extent.z = min_extent

        return bbox

    # code based on carla documentation https://carla.readthedocs.io/en/latest/tuto_G_bounding_boxes/
    def get_object_detection_annotations(self,camera,world,vehicle,carla_config):
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

        image_w = int(carla_config['ego_vehicle_settings']['camera_width'])
        image_h = int(carla_config['ego_vehicle_settings']['camera_height'])
        image_fov = int(carla_config['ego_vehicle_settings']['camera_fov'])
        listed_classes = list(carla_config['dataset_settings']['object_annotations_classes'])

        K = self.build_projection_matrix(image_w, image_h, image_fov)

        vehicles = ['vehicle.dodge.charger_2020','vehicle.dodge.charger_police','vehicle.dodge.charger_police_2020','vehicle.ford.crown','vehicle.ford.mustang','vehicle.jeep.wrangler_rubicon','vehicle.lincoln.mkz_2017','vehicle.lincoln.mkz_2020'
                    ,'vehicle.mercedes.coupe','vehicle.mercedes.coupe_2020','vehicle.micro.microlino','vehicle.mini.cooper_s','vehicle.mini.cooper_s_2021','vehicle.nissan.micra','vehicle.nissan.patrol','vehicle.nissan.patrol_2021',
                    'vehicle.seat.leon','vehicle.tesla.model3','vehicle.toyota.prius','vehicle.audi.a2','vehicle.audi.etron','vehicle.audi.tt','vehicle.bmw.grandtourer','vehicle.chevrolet.impala','vehicle.citroen.c3']
        trucks = ['vehicle.carlamotors.carlacola','vehicle.carlamotors.european_hgv','vehicle.carlamotors.firetruck','vehicle.tesla.cybertruck','vehicle.ford.ambulance','vehicle.mercedes.sprinter','vehicle.volkswagen.t2','vehicle.volkswagen.t2_2021']
        buses = ['vehicle.mitsubishi.fusorosa']
        motorcycles = ['vehicle.harley-davidson.low_rider','vehicle.kawasaki.ninja','vehicle.vespa.zx125','vehicle.yamaha.yzf']
        bikes = ['vehicle.bh.crossbike','vehicle.diamondback.century','vehicle.gazelle.omafiets']

        if 'rider' in listed_classes:
            detected_bboxes,detected_bboxes_names = self.bbox_from_mask("rider",carla_config)
        else:
            detected_bboxes = []
            detected_bboxes_names = []

        bboxes_list = []

        for npc in world.get_actors():
            if npc.id != vehicle.id:
                type = None
                if npc.type_id.startswith('walker.pedestrian') and 'person' in listed_classes:
                    type = 'person'
                elif npc.type_id in vehicles and 'vehicle' in listed_classes:
                    type = 'vehicle'
                elif npc.type_id in trucks and 'truck' in listed_classes:
                    type = 'truck'
                elif npc.type_id in buses and 'bus' in listed_classes:
                    type = 'bus'
                elif npc.type_id in motorcycles and 'motorcycle' in listed_classes:
                    type = 'motorcycle'
                elif npc.type_id in bikes and 'bicycle' in listed_classes:
                    type = 'bicycle'
                else:
                    continue

                bb = self.get_bounding_box(npc)
                dist = npc.get_transform().location.distance(vehicle.get_transform().location)

                if dist < 70:
                    forward_vec = vehicle.get_transform().get_forward_vector()
                    ray = npc.get_transform().location - vehicle.get_transform().location

                    if forward_vec.dot(ray) > 1:
                        x_max = -10000
                        x_min = 10000
                        y_max = -10000
                        y_min = 10000
                        verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                        for vert in verts:
                            p = self.get_image_point(vert, K, world_2_camera)
                            if p[0] > x_max:
                                x_max = p[0]
                            if p[0] < x_min:
                                x_min = p[0]
                            if p[1] > y_max:
                                y_max = p[1]
                            if p[1] < y_min:
                                y_min = p[1]

                        if x_min > 0 and x_max < image_w and y_min > 0 and y_max < image_h:
                            if self.is_valid_bbox([x_min, y_min, x_max, y_max], self.gt_labels,type,dict(carla_config['dataset_settings']['object_class_numpixel_threshold']),dict(carla_config['dataset_settings']['object_class_numpixel_zero_threshold'])):
                                bboxes_list.append([int(x_min), int(y_min), int(x_max), int(y_max)])
                                detected_bboxes.append([int(x_min), int(y_min), int(x_max), int(y_max)])
                                detected_bboxes_names.append(type)

        bounding_box_set = []
        bbnames = []
        if 'traffic_light' in listed_classes:
            bounding_box_set = world.get_level_bbs(carla.CityObjectLabel.TrafficLight)
            for n in range(len(bounding_box_set)):
                bbnames.append("traffic_light")
        if 'traffic_sign' in listed_classes:
            bounding_box_set.extend(world.get_level_bbs(carla.CityObjectLabel.TrafficSigns))
            for n in range(len(world.get_level_bbs(carla.CityObjectLabel.TrafficSigns))):
                bbnames.append("traffic_signs")
        if len(bounding_box_set) > 0:
            for bb in range(len(bounding_box_set)):
                if bounding_box_set[bb].location.distance(vehicle.get_transform().location) < 50:
                    forward_vec = vehicle.get_transform().get_forward_vector()
                    ray = bounding_box_set[bb].location - vehicle.get_transform().location
                    if forward_vec.dot(ray) > 1:
                        x_max = -10000
                        x_min = 10000
                        y_max = -10000
                        y_min = 10000
                        verts = [v for v in bounding_box_set[bb].get_world_vertices(carla.Transform())]
                        for vert in verts:
                            p = self.get_image_point(vert, K, world_2_camera)
                            if p[0] > x_max:
                                x_max = p[0]
                            if p[0] < x_min:
                                x_min = p[0]
                            if p[1] > y_max:
                                y_max = p[1]
                            if p[1] < y_min:
                                y_min = p[1]


                        if x_min > 0 and x_max < image_w and y_min > 0 and y_max < image_h:
                            if self.is_valid_bbox([x_min, y_min, x_max, y_max], self.gt_labels,bbnames[bb],dict(carla_config['dataset_settings']['object_class_numpixel_threshold']),dict(carla_config['dataset_settings']['object_class_numpixel_zero_threshold'])):
                                bboxes_list.append([int(x_min), int(y_min), int(x_max), int(y_max)])
                                detected_bboxes.append([int(x_min), int(y_min), int(x_max), int(y_max)])
                                detected_bboxes_names.append(type)
        return detected_bboxes,detected_bboxes_names



