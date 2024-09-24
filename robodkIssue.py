import os
import numpy as np
from robodk.robolink import *      # RoboDK's API
from robodk.robomath import *      # Math toolbox for robots
import open3d as o3d

RDK = Robolink()

SCRIPT_PATH = os.path.dirname(__file__)
BRICK_PATH = SCRIPT_PATH + '/brick.stl'

brick = RDK.AddFile(BRICK_PATH)
brick.setName('brick')
brick.setPose(Fanuc_2_Pose([1500, 0, -1210, 0, 0, 0]))
brick.Scale([1000, 1000, 1000])

cameraFrame = RDK.AddFrame('Camera Frame')
cameraFrame.setPose(Fanuc_2_Pose([1500, 0, 100, 0, -180, -90]))
cameraFrameX = RDK.AddFrame('Camera Frame X')
cameraFrameX.setPose(Fanuc_2_Pose([1350, 550, 100, 0, -180, -90]))

# Use one of this frames:
camera_frames = cameraFrameX
# camera_frames = cameraFrame

SHOT = True
VISUALISE_MESH = True

if SHOT :
        #----------------------------------
        # You might need to play arround these settings depending on the object/setup
        O3D_NORMALS_K_SIZE = 100
        O3D_MESH_POISSON_DEPTH = 9
        O3D_MESH_DENSITIES_QUANTILE = 0.05
        O3D_DISPLAY_POINTS = True
        O3D_DISPLAY_WIREFRAME = False

        #----------------------------------
        # Get the simulated camera from RoboDK

        cam_item = RDK.Item('Lidar', ITEM_TYPE_CAMERA)
        if not cam_item.Valid():
            print("Lidar simulator camera not found, try to add")
            cam_item = RDK.Cam2D_Add(camera_frames, 'FOCAL_LENGTH=6 FOV=30 FAR_LENGTH=5000 SIZE=960x540')
            cam_item.setName('Lidar')
            print("Lidar simulator camera added")
        cam_item.setParam('Open', 1)

        #----------------------------------
        # Retrieve camera settings / camera matrix
        def settings_to_dict(settings):
            if not settings:
                return {}
            settings_dict = {}
            settings_list = [setting.split('=') for setting in settings.strip().split(' ')]
            for setting in settings_list:
                key = setting[0].upper()
                val = setting[-1]

                if key in ['FOV', 'PIXELSIZE', 'FOCAL_LENGTH', 'FAR_LENGTH']:
                    val = float(val)
                elif key in ['SIZE', 'ACTUALSIZE', 'SNAPSHOT']:
                    w, h = val.split('x')
                    val = (int(w), int(h))
                elif key == val.upper():
                    val = True  # Flag

                settings_dict[key] = val

            return settings_dict
        
        cam_settings = settings_to_dict(cam_item.setParam('Settings'))
        w, h = cam_settings['SIZE']
        fy = h / (2 * np.tan(np.radians(cam_settings['FOV']) / 2))
        cam_mtx = o3d.camera.PinholeCameraIntrinsic(width=w, height=h, fx=fy, fy=fy, cx=w / 2, cy=h / 2)
        cam_pose = cam_item.getLink(ITEM_TYPE_FRAME).Pose()

        #----------------------------------------------
        # Get the depth map from socket
        depth32_socket = None
        bytes_img = RDK.Cam2D_Snapshot("", cam_item, 'DEPTH')

        #Close camera window, after shot
        RDK.Cam2D_Close()

        if isinstance(bytes_img, bytes) and bytes_img != b'':
            depth32_socket = np.frombuffer(bytes_img, dtype='>u4')
            w, h = depth32_socket[:2]
            depth32_socket = np.flipud(np.reshape(depth32_socket[2:], (h, w))).astype(np.uint32)

        # Scale it
        depth = (depth32_socket / np.iinfo(np.uint32).max) * cam_settings['FAR_LENGTH']
        depth = depth.astype(np.float32)

        #----------------------------------------------
        # Convert to point cloud, approximate mesh
        pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(depth), cam_mtx)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])  # Align with camera view
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(O3D_NORMALS_K_SIZE)
        
        if VISUALISE_MESH:
            mesh_poisson, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=O3D_MESH_POISSON_DEPTH)
            vertices_to_remove = densities < np.quantile(densities, O3D_MESH_DENSITIES_QUANTILE)
            mesh_poisson.remove_vertices_by_mask(vertices_to_remove)
            mesh_poisson.paint_uniform_color([0.5, 0.5, 0.5])
            o3d.visualization.draw_geometries([pcd, mesh_poisson] if O3D_DISPLAY_POINTS else [mesh_poisson], mesh_show_back_face=True, mesh_show_wireframe=O3D_DISPLAY_WIREFRAME)