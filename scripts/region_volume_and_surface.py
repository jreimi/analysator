import logging
import sys
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import analysator as pt


def box_mask(f, cellIDs, marginal=[150e6, 50e6, 50e6, 50e6, 50e6, 50e6]):
    """
    Crops simulation box for calculations, output flags outside of cropped box will be 0.

        :param f: a VlsvReader
        :param cellIDs: cellIDs to be masked
        :kword marginal: 6-length list of wanted marginal lengths from mesh edges in meters [negx, negy, negz, posx, posy, posz]
        :returns: 0/1 mask in input order of cellIDs.
    """
    [xmin, ymin, zmin, xmax, ymax, zmax] = f.get_spatial_mesh_extent()
    coords = f.get_cell_coordinates(cellIDs)
    res = np.zeros((len(coords)), dtype=bool)

    for i,coord in enumerate(coords):
        if np.any((coord[0] < xmin+marginal[0], coord[0] > xmax-marginal[3],
                    coord[1] < ymin+marginal[1], coord[1] > ymax-marginal[4],
                    coord[2] < zmin+marginal[2], coord[2] > zmax-marginal[5])):
            res[i] = 0
        else:
            res[i] = 1
    return res


def make_volume_and_surface(
    f0, f1,
    SDF_filen,
    surf_filen,
    condition_dicts,
    variable_dict={},
    volume_name="volume",
    convex=False):
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
        combined. The default is ``"or"``, meaning that the dictionary condition for a cell is
        satisfied if at least one condition is fulfilled. If set to ``"and"``,
        all conditions in the dictionary must be fulfilled.

    ``flag_type``
        Determines how the result of the dictionary is represented. The
        default ``"01"`` produces a binary flag (0 or 1), where 1 means that
        the dictionary conditions are satisfied for a cell. ``"fraction"`` produces the
        fraction of conditions that are fulfilled.

    ``flag_threshold``
        Required when ``flag_type="fraction"``. Specifies the minimum
        fraction of fulfilled conditions required for the dictionary conditions to be
        considered satisfied for a cell.

    After all condition dictionaries have been evaluated, only cells that have a positive (1)
    flag for every dictionary are included in the final volume.

    :param f0: VlsvReader object for L0 VLSV bulk file.
    :param f1: VlsvReader object for L1 VLSV bulk file.
    :param SDF_filen: Path and filename of the .vlsv file in which the
        resulting signed distance function is saved.
    :param surf_filen: Path and filename of the .vtp file in which the
        resulting VTK polydata surface is saved.
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
    """

    cellids = f0.read_variable("CellID")
    volname_init = volume_name+"_initial"

    # get all vlsvgrid points
    query_points = f0.get_cell_coordinates(cellids)
    index = int(f0.read_parameter("time"))
    final_flags = -1

    vlsv_writer = pt.vlsvfile.VlsvWriter(f1, SDF_filen)
    vlsv_writer.copy_variables_list(f1, ["CellID", "vg_rho"])


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
                    variable_data[var] = f0.read_variable(name=var, cellids=-1) # var name as is
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
                        variable_data[var] = f0.read_variable(
                            name=altvar, cellids=-1, operator=var[1])
                        altinfo(oldvar, altvar)
                    except:
                        errormsg(oldvar, altvar)
            else:
                try:
                    variable_data[var] = f0.read_variable(name=var[0], cellids=-1, operator=var[1])
                except:
                    # population/non-population var to try if reading given variable fails
                    oldvar = var[0]

                    if oldvar.startswith("proton/"):
                        altvar = oldvar[7:]
                    else:
                        altvar = "proton/"+oldvar

                    errormsg_alt(oldvar, altvar)

                    try:
                        variable_data[var] = f0.read_variable(name=altvar, cellids=-1, operator=var[1])
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

    # save initial flags to vlsv
    vlsv_writer.write_variable_info(
        pt.calculations.VariableInfo(final_flags, volname_init, "-", latex="", latexunits=""), "SpatialGrid", 1,)

    # now that we have the flags, let's move to vtk
    # add volume to vtk grid
    vtkreader = pt.vlsvfile.VlsvVtkReader()
    vtkreader.SetReader(f1)
    f1.add_cached_variable(final_flags, volname_init)
    vtkreader.Update()
    vtkreader.addArrayFromVlsv("cellid")
    vtkreader.addArrayFromVlsv(volname_init)
    vtkreader.Modified()
    vtkreader.Update()


    def check_surface(surface):
        try:
            try:
                surface.Update()
            except:
                print("ah")
            print("Points:", surface.GetOutput().GetNumberOfPoints())
            print("Cells:", surface.GetOutput().GetNumberOfCells())

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
        except:
            print("oh")





    dualgrid = vtk.vtkHyperTreeGridToDualGrid()
    dualgrid.SetInputConnection(vtkreader.GetOutputPort())
    vtkreader.Update()
    dualgrid.Update()

    #clean = vtk.vtkStaticCleanUnstructuredGrid()
    #clean.SetInputData(dualgrid.GetOutput())
    #clean.SetTolerance(0.0)
    #clean.SetToleranceIsAbsolute(False)
    #clean.Update()
    #dualgrid = clean
    #


    # separate volume
    threshold = vtk.vtkThreshold()
    threshold.SetInputArrayToProcess(
        0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, volname_init
    )
    threshold.SetInputData(dualgrid.GetOutput())
    threshold.SetUpperThreshold(1.0)
    threshold.SetLowerThreshold(1.0)
    threshold.Update()

    #geom = vtk.vtkGeometryFilter()
    #geom.SetInputData(threshold.GetOutput())
    #geom.Update()
    #surface = geom.GetOutput()
    #
    #
    surf = vtk.vtkDataSetSurfaceFilter()
    surf.SetInputConnection(threshold.GetOutputPort())
    surf.Update()
    surface = surf.GetOutput()
    vtk_polydata = surf
    print("initial surface:")

    check_surface(surf)

    if convex: #use 3D Delaunay triangulation
        clean = vtk.vtkCleanPolyData()
        clean.SetInputConnection(vtk_polydata.GetOutputPort())
        clean.ConvertLinesToPointsOff()
        clean.ConvertPolysToLinesOff()
        clean.ConvertStripsToPolysOff()
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

    else:

        clean = vtk.vtkCleanPolyData()
        clean.SetInputConnection(vtk_polydata.GetOutputPort())
        clean.ConvertLinesToPointsOff()
        clean.ConvertPolysToLinesOff()
        clean.ConvertStripsToPolysOff()
        clean.Update()
        vtk_polydata = clean
        print("after clean:")
        check_surface(vtk_polydata)

        #if volume_name != "plasmasheet":
        vtk_connectivity_filter = vtk.vtkPolyDataEdgeConnectivityFilter() #vtk.vtkPolyDataConnectivityFilter()
        vtk_connectivity_filter.SetInputConnection(vtk_polydata.GetOutputPort())
        vtk_connectivity_filter.SetExtractionModeToLargestRegion()
        vtk_connectivity_filter.Update()
        vtk_polydata = vtk_connectivity_filter
        print("after edge connectivity:")
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

    #writer = vtk.vtkXMLPolyDataWriter()
    #writer.SetInputConnection(vtk_polydata.GetOutputPort())
    #writer.SetFileName(outdir+"after_smoothing_"+str(t)+".vtp")
    #writer.Write()
    #

    vtk_normals_filter = vtk.vtkPolyDataNormals()
    vtk_normals_filter.SetInputConnection(vtk_polydata.GetOutputPort())
    vtk_normals_filter.ComputeCellNormalsOn() #on
    vtk_normals_filter.ComputePointNormalsOff() #off
    vtk_normals_filter.AutoOrientNormalsOff() #off
    vtk_normals_filter.ConsistencyOff() #off
    vtk_normals_filter.SplittingOff()
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

    # save surface to file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputConnection(vtk_area_filter.GetOutputPort())
    writer.SetFileName(surf_filen)
    writer.Write()

    # make signed distance function of the final surface volume
    implicitPolyDataDistance = vtk.vtkImplicitPolyDataDistance()
    implicitPolyDataDistance.SetInput(vtk_area_filter.GetOutput())
    sdf = np.zeros(len(query_points))
    for i, coord in enumerate(query_points):
        sdf[i] = implicitPolyDataDistance.EvaluateFunction(coord)

    #if volume_name=="plasmasheet": # plasmasheet gets a lot of bad normals so sign of SDF is sometimes wrong, fix outside magnetopause to always be outside plasmasheet
    #    mpause_SDF = variable_dict["SDF_magnetopause"]
    #    original_sdf = sdf
    #    sdf = np.where(mpause_SDF > 0, np.abs(original_sdf), original_sdf) # make plasmasheet SDF positive where mpause SDF is positive

    # save the SDF to vlsv file
    vlsv_writer.write_variable_info(
        pt.calculations.VariableInfo(sdf, "SDF", "-", latex="", latexunits=""), "SpatialGrid", 1,)


def main():

    # command line arguments: bulkfile index, region where region is one of:
        # "magnetopause", "closed", "plasmasheet", "lobes", "magnetopause_convex"

    if len(sys.argv) == 2:
        timeid = int(sys.argv[1])
        areaname = "magnetopause"
    elif len(sys.argv) == 3:
        timeid = int(sys.argv[1])
        areaname = sys.argv[2]
    else:
        areaname = "magnetopause"
        timeid = 1400

    #print(timeid)


    outdir = "/turso/group/spacephysics/vlasiator/data/L1/3D/FHA/region_ids/"+areaname+"/"
   # for now variables for 0000700-0001000 need to be read from L0 (no vlsvcache available so no variables from L1 but vtkvslvinterface needs L1 not L0?)
    datafile = "/home/group/spacephysics/vlasiator/data/L0/3D/FHA/bulk1/bulk1.{:07d}.vlsv".format(timeid)
    datafile_L1 = "/home/group/spacephysics/vlasiator/data/L1/3D/FHA/bulk1/bulk1.{:07d}.vlsv".format(timeid)



    f = pt.vlsvfile.VlsvReader(datafile)
    f1 = pt.vlsvfile.VlsvReader(datafile_L1)
    SDF_fn = outdir + "FHA_{:07d}_".format(timeid) + areaname + "_SDF.vlsv"
    surf_fn = outdir + "FHA_{:07d}_".format(timeid) + areaname + "_surface.vtp"

    variable_dict = {}
    condition_dict = []

    # make the surface and volume from conditions
    if areaname=="magnetopause": # magnetopause

        conditions = {
            "vg_rho": [None, 500000], # 0.5*sw
            "vg_connection": 0.0,
        }

        conditions["operator"] = "or"
        condition_dict.append(conditions)

        make_volume_and_surface(
            f, f1,
            SDF_fn,
            surf_fn,
            condition_dict,
            variable_dict,
            volume_name=areaname)


    elif areaname == "magnetopause_convex":
        conditions = {
             "vg_beta_star": [0.0, 0.5]}
        condition_dict.append(conditions)
        make_volume_and_surface(
            f, f1,
            SDF_fn,
            surf_fn,
            condition_dict, # can be a list of dictionaries
            variable_dict,
            volume_name=areaname,
            convex = True)



    elif areaname == "closed":
        condition_dict = {
            "vg_connection": 0,
        }

        make_volume_and_surface(
            f, f1,
            SDF_fn,
            surf_fn,
            condition_dict, # can be a list of dictionaries
            variable_dict,
            volume_name=areaname)

    elif areaname == "lobes": # inside magnetopause, outside plasmasheet and closed-closed fieldlines

        SDF_mpause_fn = "/turso/group/spacephysics/vlasiator/data/L1/3D/FHA/region_ids/magnetopause/FHA_{:07d}_".format(timeid) + "magnetopause_SDF.vlsv"
        f_mp =  pt.vlsvfile.VlsvReader(SDF_mpause_fn)
        variable_dict["SDF_magnetopause"] = f_mp.read_variable("SDF")
        conditions_SDF = {"SDF_magnetopause": [None, 0.0]}

        SDF_closed_fn = "/turso/group/spacephysics/vlasiator/data/L1/3D/FHA/region_ids/closed/FHA_{:07d}_".format(timeid) + "closed_SDF.vlsv"
        f_closed =  pt.vlsvfile.VlsvReader(SDF_closed_fn)
        variable_dict["SDF_closed"] = f_closed.read_variable("SDF")
        conditions_SDF2 = {"SDF_closed": [0.0, None]}

        SDF_plasmasheet_fn = "/turso/group/spacephysics/vlasiator/data/L1/3D/FHA/region_ids/plasmasheet/FHA_{:07d}_".format(timeid) + "plasmasheet_SDF.vlsv"
        f_ps =  pt.vlsvfile.VlsvReader(SDF_plasmasheet_fn)
        variable_dict["SDF_plasmasheet"] = f_ps.read_variable("SDF")
        conditions_SDF3 = {"SDF_plasmasheet": [0.0, None]}

        condition_dict = [conditions_SDF, conditions_SDF2, conditions_SDF3]

        make_volume_and_surface(
            f, f1,
            SDF_fn,
            surf_fn,
            condition_dict,
            variable_dict,
            volume_name=areaname)

        # separate north and south lobes
        condition_dict_N = [conditions_SDF, conditions_SDF2, conditions_SDF3, {("vg_b_vol", "x"): [0.0, None]}]
        SDF_fn = outdir + "FHA_{:07d}_".format(timeid) + "lobe_N_SDF.vlsv"
        surf_fn = outdir + "FHA_{:07d}_".format(timeid) + "lobe_N_surface.vtp"

        make_volume_and_surface(
            f, f1,
            SDF_fn,
            surf_fn,
            condition_dict_N,
            variable_dict,
            volume_name="lobe_North")

        condition_dict_S = [conditions_SDF, conditions_SDF2, conditions_SDF3, {("vg_b_vol", "x"): [None, 0.0]}]
        SDF_fn = outdir + "FHA_{:07d}_".format(timeid) + " lobe_S_SDF.vlsv"
        surf_fn = outdir + "FHA_{:07d}_".format(timeid) + "lobe_S_surface.vtp"

        make_volume_and_surface(
            f, f1,
            SDF_fn,
            surf_fn,
            condition_dict_S,
            variable_dict,
            volume_name="lobe_South")

    elif areaname=="plasmasheet":

        conditions = {"vg_beta": [1.0, None]}
        SDF_mpause_fn = "/turso/group/spacephysics/vlasiator/data/L1/3D/FHA/region_ids/magnetopause/FHA_{:07d}_".format(timeid) + "magnetopause_SDF.vlsv"
        SDF_closed_fn = "/turso/group/spacephysics/vlasiator/data/L1/3D/FHA/region_ids/closed/FHA_{:07d}_".format(timeid) + "closed_SDF.vlsv"
        f_mp =  pt.vlsvfile.VlsvReader(SDF_mpause_fn)
        f_closed =  pt.vlsvfile.VlsvReader(SDF_closed_fn)
        variable_dict["SDF_magnetopause"] = f_mp.read_variable("SDF")
        conditions_SDF_mpause = {"SDF_magnetopause": [None, 0.0]}  # SDF < 0: inside magnetopause surface
        variable_dict["SDF_closed"] = f_closed.read_variable("SDF")
        conditions_SDF_closed = {"SDF_closed": [0.0, None]} # SDF > 0: outside closed region (excludes dayside)

        condition_dict = [conditions, conditions_SDF_mpause, conditions_SDF_closed]

        make_volume_and_surface(
            f, f1,
            SDF_fn,
            surf_fn,
            condition_dict, # can be a list of dictionaries
            variable_dict,
            volume_name=areaname)


    elif areaname == "magnetosheath": # needs work

        SDF_mpause_fn = "/turso/group/spacephysics/vlasiator/data/L1/3D/FHA/region_ids/magnetopause/FHA_{:07d}_".format(timeid) + "magnetopause_SDF.vlsv"
        f_mp =  pt.vlsvfile.VlsvReader(SDF_mpause_fn)
        variable_dict["SDF_magnetopause"] = f_mp.read_variable("SDF")
        magnetopause_conds = {"SDF_magnetopause": [0.0, None]}

        SDF_bshock_fn = "/turso/group/spacephysics/vlasiator/data/L1/3D/FHA/region_ids/bowshock/FHA_{:07d}_".format(timeid) + "bowshock_SDF.vlsv"
        f_bs =  pt.vlsvfile.VlsvReader(SDF_bshock_fn)
        variable_dict["SDF_bowshock"] = f_bs.read_variable("SDF")
        bowshock_conds = {"SDF_bowshock": [None, 0.0]}

        condition_dict = [magnetopause_conds, bowshock_conds]


        make_volume_and_surface(
            f, f1,
            SDF_fn,
            surf_fn,
            condition_dict, # can be a list of dictionaries
            variable_dict,
            volume_name=areaname)



        # vol, surf, vlsv_writer = make_volume_and_surface(f, SDF_fn, surf_fn, conditions, flag_type="fraction", condition_operator="or", volume_name=areaname, smooth_surface=True)


if __name__ == "__main__":
    main()
