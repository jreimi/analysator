"""Script for making a VTK PolyData surface and SDF from the surface.
    Function "make_volume_and_surface()" takes conditions, and surface is made as outer surface of simulation
    cells that fulfill the conditions.

    Usage example for making a surface "magnetopause" using conditions for "vg_rho" and "vg_connection"
    with region border in -x direction set to -200e6 m:

    .. code-block:: python

        conditions = {
            "vg_rho": [None, 500000],
            "vg_connection": 0.0,
            }

        conditions["operator"] = "or"
        condition_dict.append(conditions)

        magnetoshere_flags, magnetopause_SDF, magnetopause_surface = make_volume_and_surface(
                                                                            f,
                                                                            condition_dicts = condition_dict,
                                                                            volume_name="magnetopause",
                                                                            convex=False,
                                                                            crop =[-200e6, None, None, None, None, None],
                                                                            inner_boundary = False):

"""

import logging
import sys
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import analysator as pt


def box_crop(f, cellIDs, crop_edges=[-300e6, None, None, None, None, None]):
    """
    Crops simulation cells using a given rectangle, output flags outside of cropped area will be 0.

        :param f: a VlsvReader
        :param cellIDs: cellIDs to be cropped
        :param crop_edges: 6-length list of wanted crop box edges in meters: [negx, negy, negz, posx, posy, posz].
            Any can be set as None for no crop in that direction.
        :returns: 0/1 mask in input order of cellIDs.
    """
    mesh_edges = f.get_spatial_mesh_extent()
    print(mesh_edges)
    coords = f.get_cell_coordinates(cellIDs)
    res = np.ones((len(coords)), dtype=int)

    # If there are Nones, use mesh edges
    edges = crop_edges.copy()
    for i in range(len(crop_edges)):
        if crop_edges[i] is None:
            edges[i] = mesh_edges[i]

    for i,coord in enumerate(coords):
        if np.any((coord[0] < edges[0],
                    coord[0] > edges[3],
                    coord[1] < edges[1],
                    coord[1] > edges[4],
                    coord[2] < edges[2],
                    coord[2] > edges[5])):
            res[i] = 0
    return res

def find_non_manifold_edges(vtk_polydata):
    '''
    vtkFeatureEdges points don't correspond to original polydata points
     -> find non-manifold edges by manually going through cells 
        and save them with original point ids
    '''
    polydata = vtk_polydata.GetOutput()

    non_manifold_array = vtk.vtkCellArray() # Array to save non-manifold edges to be used as barrier edges
    edges_done = set()

    # look at each edge in each polydata cell
    for cell_id in range(polydata.GetNumberOfCells()):
        cell = polydata.GetCell(cell_id)

        for edge_id in range(cell.GetNumberOfEdges()):
            edge = cell.GetEdge(edge_id)
            edge_point_ids = edge.GetPointIds()

            point_0 = edge_point_ids.GetId(0)
            point_1 = edge_point_ids.GetId(1)

            # edge ids are cell-specific so not unique in whole polydata
            # point ids are unique so make new unique edge identifier from them
            edge_u_id = (min(point_0, point_1), max(point_0, point_1))

            if edge_u_id not in edges_done: # new edge
                edges_done.add(edge_u_id)

                neighbour_cells = vtk.vtkIdList()
                polydata.GetCellEdgeNeighbors(cell_id, point_0, point_1, neighbour_cells)


                # edge is non-manifold if there are more than two cells connected
                connected_cells = neighbour_cells.GetNumberOfIds()
                if connected_cells > 1: # original cell is not included in neighbour cells
                    
                    nm_line = vtk.vtkLine()
                    nm_line.GetPointIds().SetId(0, point_0)
                    nm_line.GetPointIds().SetId(1, point_1)

                    non_manifold_array.InsertNextCell(nm_line)

    # put found non-manifold edges into polydata
    non_manifold_edges = vtk.vtkPolyData()
    non_manifold_edges.SetPoints(polydata.GetPoints()) # use original polydata points
    non_manifold_edges.SetLines(non_manifold_array)
    non_manifold_edges.Modified()

    return non_manifold_edges


def check_surface(surface):
    points = surface.GetOutput().GetNumberOfPoints()
    cells = surface.GetOutput().GetNumberOfCells()
    print("Points:", points)
    print("Cells:", cells)

    if points == 0 and cells == 0:
        print("No points or cells, surface cannot be made.")
        sys.exit()

    fe = vtk.vtkFeatureEdges()
    fe.SetInputData(surface.GetOutput())
    fe.NonManifoldEdgesOn()
    fe.BoundaryEdgesOff()
    fe.FeatureEdgesOff()
    fe.ManifoldEdgesOff()
    fe.Update()
    print("non-manifold:", fe.GetOutput().GetNumberOfCells())


    fe2 = vtk.vtkFeatureEdges()
    fe2.SetInputData(surface.GetOutput())
    fe2.BoundaryEdgesOn()
    fe2.NonManifoldEdgesOff()
    fe2.FeatureEdgesOff()
    fe2.ManifoldEdgesOff()
    fe2.Update()
    print("boundary:", fe2.GetOutput().GetNumberOfCells())




def make_volume_and_surface(
    f,
    condition_dicts,
    variable_dict={},
    volume_name="volume",
    convex=False,
    crop=False,
    inner_boundary=False,
    return_sdf = True):
    """
    Create a volume and its outer surface from VLSV data.

    The volume consists of VLSV simulation cells that satisfy the specified
    variable-value conditions. The resulting surface is saved as VTK polydata to .vtp file,
    and volume is saved to .vlsv file as signed distances ("SDF") from all simulation cells to surface.

    Conditions are specified as one or more dictionaries. Each dictionary defines one set of conditions
    that is evaluated for every cell. If multiple condition dictionaries are provided, a cell must
    satisfy all dictionaries to be included in the final volume.

    Conditions can be specified for variables available through the VlsvReader.
    Variables that are not available through the reader can instead be provided through variable_dict.
    The keys in variable_dict must match the corresponding variable names used in the condition
    dictionaries.

    Each condition dictionary can contain the following keys:

    ``operator``
        Determines how the individual conditions within the dictionary are
        combined. The default is "or", meaning that the dictionary condition for a cell is
        satisfied if at least one condition is fulfilled. If set to "and",
        all conditions in the dictionary must be fulfilled.

    ``flag_type``
        Determines how the result of the dictionary is represented. The
        default "01" produces a binary flag (0 or 1), where 1 means that
        the dictionary conditions are satisfied for a cell. "fraction" produces the
        fraction of conditions that are fulfilled.

    ``flag_threshold``
        Required when ``flag_type="fraction"``. Specifies the minimum
        fraction of fulfilled conditions required for the dictionary conditions to be
        considered satisfied for a cell.

    After all condition dictionaries have been evaluated, only cells that have a positive (1)
    flag for every dictionary are included in the final volume.

    :param f: VlsvReader object for L1 VLSV bulk file.
    :param condition_dicts: A condition dictionary or a list of condition
        dictionaries defining the variable-value conditions used to select
        cells for the volume.
    :param variable_dict: Optional dictionary containing values for variables
        that cannot be read using VlsvReader. The dictionary keys must
        match variable names used in any condition dictionary. Defaults to an
        empty dictionary.
    :param volume_name: Name of the volume. The corresponding surface is
        named "volume_name" in the .vtp file. Defaults to "volume".
    :param convex: If ``True``, use VTK's vtkDelaunay3D  to create the convex
        hull of the selected volume, producing an outer convex surface.
        Defaults to ``False``.
    :param crop: Defaults to ``False``. A 6-length list of distances in meters
        can be given to crop initial region at some axis value.
        Axes are [negx, negy, negz, posx, posy, posz], and any can be set as
        None for no crop in that direction.
    :param inner_boundary: Whether to exclude the inner boundary at some distance
        from Earth. Defaults to ``False``. If ``True``, or float, initial
        flags are set to 0 within ... m (``True``) or the given distance
        (float, in meters) from the Earth.
    :returns: TODO
    """

    cellids = f.read_variable("CellID")
    volname_init = volume_name+"_initial"

    # get all vlsvgrid points
    query_points = f.get_cell_coordinates(cellids)
    index = int(f.read_parameter("time"))
    final_flags = -1

    # error messages
    def errormsg_alt(varstr, altvar):
            logging.warning("{0} could not be read. Trying to read {1} instead".format(varstr, altvar))

    def errormsg(varstr,altvar):
        logging.error("{0} could not be read, {1} will be ignored".format(altvar, varstr))

    def altinfo(varstr, altvar):
        logging.info("{0} was found, using {0} instead of {1}".format(altvar, varstr))


    if isinstance(condition_dicts, dict): # just one
        condition_dicts = [condition_dicts]


    for condition_dict in condition_dicts:
        flags = np.zeros(len(cellids))
        variable_data = {}

        conditions = list(condition_dict.keys()) # keep track of filled conditions

        if "operator" in conditions:
            condition_operator = condition_dict["operator"]
            conditions.remove("operator")
        else:
            condition_operator = "or"

        if "flag_type" in conditions:
            flag_type = condition_dict["flag_type"]
            conditions.remove("flag_type")
        else:
            flag_type = "01"

        if "flag_threshold" in conditions:
            flag_threshold = condition_dict["flag_threshold"]
            conditions.remove("flag_threshold")

        # read variables that were given as arrays
        if isinstance(variable_dict, dict):
            for value_var in set(condition_dict.keys()) & set(variable_dict.keys()):
                variable_data[value_var] = variable_dict[value_var]
                conditions.remove(value_var)

        # read other variables from datafile
        for var in conditions: # variables to be read straight from vlsvreader # Must be a better way to do this
            if isinstance(var, str):  # no operator
                try:
                    variable_data[var] = f.read_variable(name=var, cellids=-1) # var name as is
                except:
                    # population/non-population var to try if reading given variable fails
                    oldvar = var
                    print(oldvar)
                    if oldvar.startswith("proton/"):
                        altvar = oldvar[7:]
                    else:
                        altvar = "proton/"+oldvar

                    errormsg_alt(oldvar, altvar)

                    try:
                        variable_data[var] = f.read_variable(
                            name=altvar, cellids=-1, operator=var[1])
                        altinfo(oldvar, altvar)
                    except:
                        errormsg(oldvar, altvar)
            else:
                try:
                    variable_data[var] = f.read_variable(name=var[0], cellids=-1, operator=var[1])
                except:
                    # population/non-population var to try if reading given variable fails
                    oldvar = var[0]

                    if oldvar.startswith("proton/"):
                        altvar = oldvar[7:]
                    else:
                        altvar = "proton/"+oldvar

                    errormsg_alt(oldvar, altvar)

                    try:
                        variable_data[var] = f.read_variable(name=altvar, cellids=-1, operator=var[1])
                        altinfo(oldvar, altvar)
                    except:
                        errormsg(oldvar, altvar)


        print("Conditions:")
        print(condition_dict)
        result_mask = np.zeros((len(variable_data.keys()), len(cellids)))


        ## make a 1/0 mask of cells where conditions are true for all variables that were read
        for i, variable in enumerate(variable_data.keys()):
            condition = condition_dict[variable]
            data = variable_data[variable]

            if isinstance(condition, float) or isinstance(condition, int):  # single value, exact
                mask = np.where(np.isclose(data, condition), 1, 0)
            elif isinstance(condition, tuple):  # single value with tolerance
                mask = np.where(np.isclose(data, condition[0], rtol=condition[1]), 1, 0)
            else:
                if condition[0] is None:
                    mask = np.where((data <= condition[1]), 1, 0)  # anywhere where value is less than or equal to
                elif condition[1] is None:
                    mask = np.where((condition[0] <= data), 1, 0)  # anywhere where value is more than or equal to
                else:
                    mask = np.where((condition[0] <= data) & (data <= condition[1]), 1, 0)  # min/max

            result_mask[i] = mask

        if (flag_type == "01" and condition_operator == "and"):  # 1 only if all viable variables are 1, 0 otherwise
            flags = np.prod(result_mask, axis=0)
            # return flags

        elif (flag_type == "01" and condition_operator == "or"):  # 1 if any of the viable variables are 1, 0 otherwise
            flags = np.where(np.sum(result_mask, axis=0) > 0, 1, 0)
            # return flags

        elif flag_type == "fraction":  # rounded fraction of viable variables that are 1
            flags = np.sum(result_mask, axis=0) / len(result_mask)
            flags = np.where(flags > flag_threshold, 1, 0)

        # now we have 01 flags for the current condition dictionary, add to final flags
        if isinstance(final_flags, np.ndarray): # flags exist from another condition dictionary
            final_flags = final_flags*flags
        else:
            final_flags = flags

    # check if there are any flags
    if isinstance(final_flags, int):
        print("No flags found, discarding timestep.")
        return 0

    # cropping the region should be done here if needed
    if crop is not False:
        crop_mask = box_crop(f, cellids, crop_edges=crop)
        final_flags = final_flags*crop_mask

    # manual inner boundary for volume
    if inner_boundary is not False:
        if inner_boundary is True:
            rad = 35.7e6 # FHA coupling radius
        elif isinstance(inner_boundary, float):
            rad = inner_boundary

        coords = f.get_cell_coordinates(cellids)
        norms = np.linalg.norm(coords, axis=1)
        final_flags[norms <= rad] = 0


    # now that we have the flags, let's move to vtk
    # add volume to vtk grid
    vtkreader = pt.vlsvfile.VlsvVtkReader()
    vtkreader.SetReader(f)
    f.add_cached_variable(final_flags, volume_name)
    vtkreader.Update()
    vtkreader.addArrayFromVlsv("cellid")
    vtkreader.addArrayFromVlsv(volume_name)
    vtkreader.Modified()
    vtkreader.Update()

    # get a dualgrid out of VlsvVtkReader
    dualgrid = vtk.vtkHyperTreeGridToDualGrid()
    dualgrid.SetInputConnection(vtkreader.GetOutputPort())
    vtkreader.Update()
    dualgrid.Update()

    # separate volume using flags
    threshold = vtk.vtkThreshold()
    threshold.SetInputArrayToProcess(
        0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, volume_name
    )
    threshold.SetInputData(dualgrid.GetOutput())
    threshold.SetUpperThreshold(1.0)
    threshold.SetLowerThreshold(1.0)
    threshold.Update()

    # make outer surface out of the thresholded volume
    surf = vtk.vtkDataSetSurfaceFilter()
    surf.SetInputConnection(threshold.GetOutputPort())
    surf.Update()
    surface = surf.GetOutput()# save surface
    if True: #tf is this?
        #init_surf_filen = outdir + "FHA_{:07d}_".format(timeid) + areaname + "_initial_surface.vtp"
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetInputData(surface) # check this
        writer.SetFileName("gradu/tests/initial_surface.vtp")
        writer.Write()

        # save SDF
        #vlsv_writer.write_variable_info(
        #    pt.calculations.VariableInfo(sdf, "vg_SDF_"+areaname, "-", latex="", latexunits=""), "SpatialGrid", 1,)

        #final_flags = flags # magnetosphere: 0=outside, 1=inside
    vtk_polydata = surf
    print("initial surface:")

    check_surface(surf)

    trisurf = vtk.vtkTriangleFilter()
    trisurf.SetInputConnection(vtk_polydata.GetOutputPort())
    trisurf.Update()
    print("After triangulation:")
    check_surface(trisurf)
    vtk_polydata = trisurf

    if convex: #use 3D Delaunay triangulation
        clean = vtk.vtkCleanPolyData()
        clean.SetInputConnection(vtk_polydata.GetOutputPort())
        clean.Update()
        vtk_polydata = clean
        print("after clean:")
        check_surface(vtk_polydata)

        delaunay_3d = vtk.vtkDelaunay3D()
        delaunay_3d.SetInputData(vtk_polydata.GetOutput())
        alpha = None
        if alpha is not None:
            delaunay_3d.SetAlpha(alpha)
            delaunay_3d.SetAlphaTets(1)
            delaunay_3d.SetAlphaLines(0)
            delaunay_3d.SetAlphaTris(0)
            delaunay_3d.SetAlphaVerts(0)
        delaunay_3d.Update()

        convex = delaunay_3d.GetOutput()
        surface = vtk.vtkDataSetSurfaceFilter()
        surface.SetInputData(convex)
        surface.Update()
        vtk_polydata = surface

        print("Delaunay surface:")
        check_surface(surface)

        # smooth the surface:
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(vtk_polydata.GetOutputPort())
        smoother.SetNumberOfIterations(5)
        smoother.BoundarySmoothingOn()
        smoother.NonManifoldSmoothingOff()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()
        vtk_polydata=smoother
        print("after smoothing:")
        check_surface(vtk_polydata)

    else: # no Delaunay, just surface of whatever flagged volume we had

        clean = vtk.vtkCleanPolyData()
        clean.SetInputConnection(vtk_polydata.GetOutputPort())
        #clean.ConvertLinesToPointsOff() # needs to be on (default)
        clean.ConvertPolysToLinesOff() # needs to be off !
        #clean.ConvertStripsToPolysOff() # does not seem to matter
        clean.Update()
        vtk_polydata = clean
        print("after clean:")
        check_surface(vtk_polydata)

        non_manifold_edges = find_non_manifold_edges(vtk_polydata)        
        
        # edge connectivity filter, outputs largest region using edge connectivity
        # and uses non-manifold edges as barrier edges (note: some non-manifold edges may remain)
        vtk_connectivity_filter = vtk.vtkPolyDataEdgeConnectivityFilter() 
        vtk_connectivity_filter.SetInputConnection(vtk_polydata.GetOutputPort())
        vtk_connectivity_filter.SetSourceData(non_manifold_edges) 
        vtk_connectivity_filter.BarrierEdgesOn()
        vtk_connectivity_filter.SetExtractionModeToLargestRegion()
        vtk_connectivity_filter.Update()
        vtk_polydata = vtk_connectivity_filter
        print("after connectivity:")
        check_surface(vtk_polydata)

        # smooth the surface:
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(vtk_polydata.GetOutputPort())
        smoother.SetNumberOfIterations(30)
        smoother.BoundarySmoothingOn()
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()
        vtk_polydata=smoother
        print("after smoothing:")
        check_surface(vtk_polydata)


    vtk_normals_filter = vtk.vtkTriangleMeshPointNormals()
    vtk_normals_filter.SetInputConnection(vtk_polydata.GetOutputPort())
    #vtk_normals_filter.ComputeCellNormalsOn() #on
    #vtk_normals_filter.ComputePointNormalsOff() #off
    #vtk_normals_filter.AutoOrientNormalsOff() #off
    #vtk_normals_filter.ConsistencyOff() #off
    #vtk_normals_filter.SplittingOff()
    vtk_normals_filter.Update()
    vtk_polydata = vtk_normals_filter
    print("after normals:") # should not change with splitting off but anyway
    check_surface(vtk_polydata)

    # cell centers
    vtk_center_filter = vtk.vtkCellCenters()
    vtk_center_filter.SetInputData(vtk_polydata.GetOutput())
    vtk_center_filter.Update()
    # centers need to be added manually so that polydata doesn't lose cells (?)
    centerArray = vtk_center_filter.GetOutput().GetPoints().GetData()
    centerArray.SetName("Centers")
    vtk_polydata.GetOutput().GetCellData().AddArray(centerArray)

    # cell areas (as "Quality")
    vtk_area_filter = vtk.vtkMeshQuality()
    vtk_area_filter.SetInputConnection(vtk_polydata.GetOutputPort())
    vtk_area_filter.SetTriangleQualityMeasureToArea()
    vtk_area_filter.SetQuadQualityMeasureToArea()
    vtk_area_filter.Update()

    if return_sdf:
        # make signed distance function of the final surface volume
        implicitPolyDataDistance = vtk.vtkImplicitPolyDataDistance()
        implicitPolyDataDistance.SetInput(vtk_area_filter.GetOutput())
        sdf = np.zeros(len(query_points))
        for i, coord in enumerate(query_points):
            sdf[i] = implicitPolyDataDistance.EvaluateFunction(coord)
    else:
        sdf = None

    # return region flags (for .vlsv), sdf to surface (for .vlsv), surface polydata (for .vtp)
    return final_flags, sdf, vtk_area_filter.GetOutput()


def main():

    # command line arguments: bulkfile index

    if len(sys.argv) == 2:
        timeid = int(sys.argv[1])
    else:
        timeid = 1400

    #print(timeid)

    outdir = "gradu/regions/"
    datafile_L1 = "/home/group/spacephysics/vlasiator/data/L1/3D/FHA/bulk1/bulk1.{:07d}.vlsv".format(timeid)

    f = pt.vlsvfile.VlsvReader(datafile_L1)
    SDF_fn = outdir + "SDFs/FHA_{:07d}".format(timeid) + "_regions.vlsv"

    # .vlsv file for saving flags and SDFs
    vlsv_writer = pt.vlsvfile.VlsvWriter(f, SDF_fn)
    vlsv_writer.copy_variables_list(f, ["CellID", "vg_rho", "vg_b_vol"])



    # possible inner areas:
    do = {"inner_boundary": False,
        "dayside_closed": False,
        "nightside_closed": False,
        "plasmasheet": False,
        "lobes": False}


    # improvement could be made that all initial flags go to single dualgrid and separate from there 
    # but that would require none of the regions depending on the others

    ########################################################################################
    # MAGNETOPAUSE
    areaname="magnetopause"
    variable_dict = {}
    condition_dict = []

    conditions = {
        "vg_rho": [None, 500000], # 0.5*sw
        "vg_connection": 0.0,
    }

    conditions["operator"] = "or"
    condition_dict.append(conditions)

    magnetopause_flags, magnetopause_sdf, magnetopause_surface = make_volume_and_surface(
                                        f,
                                        condition_dict,
                                        variable_dict,
                                        volume_name=areaname,
                                        crop = [-300e6, None, None, None, None, None],
                                        inner_boundary = False)

    # save surface
    surf_filen = outdir + areaname +"/FHA_{:07d}_".format(timeid) + areaname + "_surface.vtp"
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(magnetopause_surface)
    writer.SetFileName(surf_filen)
    writer.Write()

    # save SDF
    vlsv_writer.write_variable_info(
        pt.calculations.VariableInfo(magnetopause_sdf, "vg_SDF_"+areaname, "-", latex="", latexunits=""), "SpatialGrid", 1,)

    final_flags = magnetopause_flags # magnetosphere: 0=outside, 1=inside


    ########################################################################################
    # INNER BOUNDARY (coupling radius at 35.7e6 m (≈ 5.6 RE))

    if do["inner_boundary"]:
        areaname = "inner_boundary"
        condition_dict = {
            "radius": [None, 3.57e7] # where radius is less than inner boundary radius
        }

        cells = f.read_variable("CellID")
        coords = f.get_cell_coordinates(cells)
        radii = np.linalg.norm(coords, axis=1)
        variable_dict = {"radius": radii}

        inner_flags, inner_sdf, inner_surface = make_volume_and_surface(
                                            f,
                                            condition_dict,
                                            variable_dict,
                                            volume_name=areaname,
                                            crop = [-300e6, None, None, None, None, None],
                                            inner_boundary=False,
                                            return_sdf=False)

        # save surface
        surf_filen = outdir + areaname +"/FHA_{:07d}_".format(timeid) + areaname + "_surface.vtp"
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetInputData(inner_surface) 
        writer.SetFileName(surf_filen)
        writer.Write()

        # inner boundary inside region flag: 2
        final_flags[inner_flags == 1] = 2

    ########################################################################################
    # DAYSIDE
    if do["dayside_closed"]:
        areaname = "dayside_closed"
        variable_dict = {}
        condition_dict = {
            "vg_connection": 0,
            ("vg_connection_coordinates_bw_x", "x"): [0, None],
            ("vg_connection_coordinates_fw_x", "x"): [0, None] 
        }


        dayside_flags,dayside_sdf, dayside_surface = make_volume_and_surface(
                                            f,
                                            condition_dict,
                                            variable_dict,
                                            volume_name=areaname,
                                            crop = [-300e6, None, None, None, None, None],
                                            inner_boundary = True,
                                            return_sdf=True)

        # save surface
        surf_filen = outdir + areaname +"/FHA_{:07d}_".format(timeid) + areaname + "_surface.vtp"
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetInputData(dayside_surface) # check this
        writer.SetFileName(surf_filen)
        writer.Write()

        # save SDF
        vlsv_writer.write_variable_info(
            pt.calculations.VariableInfo(dayside_sdf, "vg_SDF_"+areaname, "-", latex="", latexunits=""), "SpatialGrid", 1,)

        # dayside closed region flag: 3
        final_flags[dayside_flags == 1] = 3

    ########################################################################################
    # CLOSED REGION NIGHTSIDE
    if do["nightside_closed"]:
        areaname = "nightside_closed"
        variable_dict = {}
        condition_dict = {
            "vg_connection": 0,
            ("vg_connection_coordinates_bw_x", "x"): [None, 0],
            ("vg_connection_coordinates_fw_x", "x"): [None, 0] 
        }

        nightside_flags, nightside_sdf, nightside_surface = make_volume_and_surface(
                                            f,
                                            condition_dict,
                                            variable_dict,
                                            volume_name=areaname,
                                            crop = [-300e6, None, None, None, None, None],
                                            inner_boundary = True,
                                            )

        # save surface
        surf_filen = outdir + areaname +"/FHA_{:07d}_".format(timeid) + areaname + "_surface.vtp"
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetInputData(nightside_surface)
        writer.SetFileName(surf_filen)
        writer.Write()

        # save SDF
        vlsv_writer.write_variable_info(
            pt.calculations.VariableInfo(nightside_sdf, "vg_SDF_"+areaname, "-", latex="", latexunits=""), "SpatialGrid", 1,)

        # nightside closed region flag: 4
        final_flags[closed_flags == 1] = 4

    ########################################################################################
    # PLASMASHEET
    if do["plasmasheet"]:
        areaname="plasmasheet"
        variable_dict = {}
        condition_dict = {}

        conditions = {"vg_beta": [1.0, None]}

        variable_dict["SDF_magnetopause"] = magnetopause_sdf
        conditions_SDF_mpause = {"SDF_magnetopause": [None, 0.0]}  # SDF < 0: inside magnetopause surface
        
        variable_dict["SDF_dayside_closed"] = dayside_sdf
        conditions_SDF_dayside = {"SDF_dayside_closed": [0.0, None]} # SDF > 0: outside closed regions
        variable_dict["SDF_nightside_closed"] = nightside_sdf
        conditions_SDF_nightside = {"SDF_nightside_closed": [0.0, None]} 

        condition_dict = [conditions, conditions_SDF_mpause, conditions_SDF_dayside, conditions_SDF_nightside]

        plasmasheet_flags, plasmasheet_sdf, plasmasheet_surface = make_volume_and_surface(
                                                                    f,
                                                                    condition_dict,
                                                                    variable_dict,
                                                                    volume_name=areaname,
                                                                    inner_boundary = True)

        # save surface
        surf_filen = outdir + areaname +"/FHA_{:07d}_".format(timeid) + areaname + "_surface.vtp"
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetInputData(plasmasheet_surface) # check this
        writer.SetFileName(surf_filen)
        writer.Write()

        # save SDF
        vlsv_writer.write_variable_info(
            pt.calculations.VariableInfo(plasmasheet_sdf, "vg_SDF_"+areaname, "-", latex="", latexunits=""), "SpatialGrid", 1,)

        # plasmasheet region flag: 5
        final_flags[plasmasheet_flags == 1] = 5


    ########################################################################################
    # LOBES
    if do["lobes"]:
        areaname = "lobes" # inside magnetopause, outside plasmasheet and closed-closed fieldlines

        variable_dict = {}

        variable_dict["SDF_magnetopause"] = magnetopause_sdf
        conditions_SDF = {"SDF_magnetopause": [None, 0.0]}

        variable_dict["SDF_dayside_closed"] = dayside_sdf
        conditions_SDF_dayside = {"SDF_dayside_closed": [0.0, None]} # SDF > 0: outside closed regions
        variable_dict["SDF_nightside_closed"] = nightside_sdf
        conditions_SDF_nightside = {"SDF_nightside_closed": [0.0, None]} 

        variable_dict["SDF_plasmasheet"] = plasmasheet_SDF
        conditions_SDF3 = {"SDF_plasmasheet": [0.0, None]}


        # separate north and south lobes
        areaname = "lobe_N"
        condition_dict_N = [conditions_SDF, conditions_SDF_dayside, conditions_SDF_nightside, conditions_SDF3, {("vg_b_vol", "x"): [0.0, None]}]

        lobe_N_flags, lobe_N_sdf, lobe_N_surface = make_volume_and_surface(
                                                        f,
                                                        condition_dict_N,
                                                        variable_dict,
                                                        volume_name=areaname,
                                                        inner_boundary = True)

        # save surface
        surf_filen = outdir + areaname +"/FHA_{:07d}_".format(timeid) + areaname + "_surface.vtp"
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetInputData(lobe_N_surface)
        writer.SetFileName(surf_filen)
        writer.Write()

        # save SDF
        vlsv_writer.write_variable_info(
            pt.calculations.VariableInfo(lobe_N_sdf, "vg_SDF_"+areaname, "-", latex="", latexunits=""), "SpatialGrid", 1,)

        # Northern lobe region flag: 6
        final_flags[lobe_N_flags == 1] = 6


        areaname="lobe_S"
        condition_dict_S = [conditions_SDF, conditions_SDF2, conditions_SDF3, {("vg_b_vol", "x"): [None, 0.0]}]
        lobe_S_flags, lobe_S_sdf, lobe_S_surface = make_volume_and_surface(
                                                        f,
                                                        condition_dict_S,
                                                        variable_dict,
                                                        volume_name=areaname,
                                                        inner_boundary = True,)

        # save surface
        surf_filen = outdir + areaname +"/FHA_{:07d}_".format(timeid) + areaname + "_surface.vtp"
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetInputData(lobe_S_surface) 
        writer.SetFileName(surf_filen)
        writer.Write()

        # save SDF
        vlsv_writer.write_variable_info(
            pt.calculations.VariableInfo(lobe_S_sdf, "vg_SDF_"+areaname, "-", latex="", latexunits=""), "SpatialGrid", 1,)

        # Southern lobe region flag: 7
        final_flags[lobe_S_flags == 1] = 7

    ########################################################################################

    ## save all initial flags to single .vlsv variable (to do from SDFs)
    # 0 = outside magnetosphere
    # 1 = magnetosphere (magnetopause inner region) (should not exist in final flags if inner regions fill the magnetosphere)
    # 2 = inner boundary inner region
    # 3 = dayside closed
    # 4 = nightside closed
    # 5 = plasmasheet?
    # 6 = Northern lobe
    # 7 = Southern lobe

    vlsv_writer.write_variable_info(
        pt.calculations.VariableInfo(final_flags, "vg_regions", "-", latex="", latexunits=""), "SpatialGrid", 1,)



if __name__ == "__main__":
    main()
