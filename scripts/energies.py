import analysator as pt
import numpy as np
import matplotlib.pyplot as plt
import sys
import vtk
import logging
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk



def calculate_energies(fn, in_surface_fn, out_surface_flux_fn, in_SDF_fn, out_energy_vlsv_fn, out_path, out_csv):
    """
    Calculate energy fluxes through a surface and energies within a volume.

    The surface flux is calculated using data interpolated at the cell centers
    of the VTK surface.

    :param fn: Name of the .vlsv data file to be read by vlsvReader.
    :param in_surface_fn: Name of the input .vtp file containing the VTK
        PolyData object describing the surface.
    :param out_surface_flux_fn: Name of the .vtp output file for the calculated
        surface energy fluxes.
    :param in_SDF_fn: Name of the .vlsv input file containing the signed distance
        function SDF named "SDF".
    :param out_energy_vlsv_fn: Name of the output .vlsv file containing
        the calculated energies within the volume.
    :param out_path: Path to the directory where output files are stored.
    :param out_csv: Name of the output CSV file containing the calculated
        energy values.
    """


    ## SURFACE
    # datafile reader
    f = pt.vlsvfile.VlsvReader(file_name=fn)

    # read vtk surface
    vtk_reader = vtk.vtkXMLPolyDataReader()
    vtk_reader.SetFileName(in_surface_fn)
    vtk_reader.Update()
    vtk_polydata = vtk_reader.GetOutput()

    # make sure we have surface cell normals, areas and centres
    try:
        normals = vtk_to_numpy(vtk_polydata.GetCellData().GetArray("Normals"))
        print("normals found")
    except:
        print("Cell surface normals not found in polydata, calculating normals")
        vtk_normals_filter = vtk.vtkPolyDataNormals()
        vtk_normals_filter.SetInputData(vtk_polydata)
        vtk_normals_filter.ComputeCellNormalsOn()
        vtk_normals_filter.ComputePointNormalsOff()
        vtk_normals_filter.AutoOrientNormalsOff()
        vtk_normals_filter.ConsistencyOff()
        vtk_normals_filter.update()
        normalsArray = vtk.vtkFloatArray()
        normalsArray = vtk_normals_filter.GetOutput().GetCellData().GetNormals()
        normals = np.array(normalsArray)
        #print(normals)
    try:
        centers = vtk_to_numpy(vtk_polydata.GetCellData().GetArray("Centers"))
        print("centers found")
    except:
        print("Cell centers not found in polydata, calculating centers")
        vtk_center_filter = vtk.vtkCellCenters()
        vtk_center_filter.SetInputData(vtk_polydata)
        vtk_center_filter.Update()
        centerArray = vtk_center_filter.GetOutput().GetPoints().GetData()
        centers = np.array(centerArray)
        #print(centers)

    try:
        areas = vtk_to_numpy(vtk_polydata.GetCellData().GetArray("Quality"))
        print("areas found")
    except:
        print("Cell areas not found in polydata, calculating areas")
        vtk_area_filter = vtk.vtkMeshQuality()
        vtk_area_filter.SetInputData(vtk_polydata)
        vtk_area_filter.SetTriangleQualityMeasureToArea()
        vtk_area_filter.SetQuadQualityMeasureToArea()
        vtk_area_filter.Update()
        areas = vtk_to_numpy(vtk_area_filter.GetOutput().GetCellData().GetArray("Quality"))
        #print(areas)


    coords = np.array(centers)
    n_points = np.shape(coords)[0]

    S = f.read_interpolated_variable("vg_poynting", coords)
    rho = f.read_interpolated_variable("vg_rhom", coords)
    V_norm = f.read_interpolated_variable("vg_v", coords,  operator="magnitude")
    V = f.read_interpolated_variable("vg_v", coords)
    p_tensor = f.read_interpolated_variable("vg_ptensor", coords)
    p_diag = f.read_interpolated_variable("vg_ptensor_diagonal", coords)


    n_points = np.shape(coords)[0]
    rho = rho.reshape(n_points, 1)
    V_norm = V_norm.reshape(n_points, 1)
    p_diag_sum = np.sum(p_diag, axis=1).reshape(n_points, 1)

    # calculate energy fluxes through surface
    kinetic = 0.5 * rho * V_norm**2 * V
    thermal = 0.5 * p_diag_sum * V
    pressure_work = (p_tensor @ V[:, :, None]).squeeze(-1)

    # Hydrodynamic and total flux
    H = kinetic + thermal + pressure_work
    K = S + H

    Normal_component_K = np.einsum('ij,ij->i', K, normals) #np.sum(K*normals, axis=1) #areas*np.einsum('ij,ij->i', K, normals)
    Normal_component_K = np.nan_to_num(Normal_component_K)
    K_cell_flux = np.nan_to_num(areas*np.einsum('ij,ij->i', K, normals))


    # also get Poynting flux, hydrodynamic flux (and hd fluxes separately, just save everything)
    Normal_component_S = np.nan_to_num(np.einsum('ij,ij->i', S, normals))
    Normal_component_H = np.nan_to_num(np.einsum('ij,ij->i', H, normals))
    Normal_component_kinetic = np.nan_to_num(np.einsum('ij,ij->i', kinetic, normals))
    Normal_component_thermal = np.nan_to_num(np.einsum('ij,ij->i', thermal, normals))
    Normal_component_pressure_work = np.nan_to_num(np.einsum('ij,ij->i', pressure_work, normals))

    S_cell_flux = np.nan_to_num(areas*np.einsum('ij,ij->i', S, normals))
    H_cell_flux = np.nan_to_num(areas*np.einsum('ij,ij->i', H, normals))
    kinetic_cell_flux = np.nan_to_num(areas*np.einsum('ij,ij->i', kinetic, normals))
    thermal_cell_flux = np.nan_to_num(areas*np.einsum('ij,ij->i', thermal, normals))
    pressure_work_cell_flux = np.nan_to_num(areas*np.einsum('ij,ij->i', pressure_work, normals))


    # save energy flux vectors to VTK polydata
    K_array = numpy_to_vtk(K)
    K_array.SetName("K")
    vtk_polydata.GetCellData().AddArray(K_array)
    vtk_polydata.Modified()
    S_array = numpy_to_vtk(S)
    S_array.SetName("S")
    vtk_polydata.GetCellData().AddArray(S_array)
    vtk_polydata.Modified()
    H_array = numpy_to_vtk(H)
    H_array.SetName("H")
    vtk_polydata.GetCellData().AddArray(H_array)
    vtk_polydata.Modified()
    kin_array = numpy_to_vtk(kinetic)
    kin_array.SetName("kinetic_flux")
    vtk_polydata.GetCellData().AddArray(kin_array)
    vtk_polydata.Modified()
    the_array = numpy_to_vtk(thermal)
    the_array.SetName("thermal_flux")
    vtk_polydata.GetCellData().AddArray(the_array)
    vtk_polydata.Modified()
    pre_array = numpy_to_vtk(pressure_work)
    pre_array.SetName("pressure_flux")
    vtk_polydata.GetCellData().AddArray(pre_array)
    vtk_polydata.Modified()

    # save energy fluxes through surface
    KFlux_array = numpy_to_vtk(Normal_component_K, deep=True)
    KFlux_array.SetName("Surface_K_Flux")
    vtk_polydata.GetCellData().AddArray(KFlux_array)
    vtk_polydata.Modified()
    SFlux_array = numpy_to_vtk(Normal_component_S, deep=True)
    SFlux_array.SetName("Surface_S_Flux")
    vtk_polydata.GetCellData().AddArray(SFlux_array)
    vtk_polydata.Modified()
    HFlux_array = numpy_to_vtk(Normal_component_H, deep=True)
    HFlux_array.SetName("Surface_H_Flux")
    vtk_polydata.GetCellData().AddArray(HFlux_array)
    vtk_polydata.Modified()
    KineticFlux_array = numpy_to_vtk(Normal_component_kinetic, deep=True)
    KineticFlux_array.SetName("Surface_kinetic_Flux")
    vtk_polydata.GetCellData().AddArray(KineticFlux_array)
    vtk_polydata.Modified()
    ThermalFlux_array = numpy_to_vtk(Normal_component_thermal, deep=True)
    ThermalFlux_array.SetName("Surface_thermal_Flux")
    vtk_polydata.GetCellData().AddArray(ThermalFlux_array)
    vtk_polydata.Modified()
    PressureFlux_array = numpy_to_vtk(Normal_component_pressure_work, deep=True)
    PressureFlux_array.SetName("Surface_pressure_Flux")
    vtk_polydata.GetCellData().AddArray(PressureFlux_array)
    vtk_polydata.Modified()

    # calcuate fluxes inwards and outwards separately
    def separate_fluxes(flux):
        out = np.where(flux > 0, flux, 0.0)
        inwards = np.where(flux < 0, -flux, 0.0)
        return out, inwards

    kinetic_escape, kinetic_injection = separate_fluxes(kinetic_cell_flux)
    thermal_escape, thermal_injection = separate_fluxes(thermal_cell_flux)
    pressure_work_escape, pressure_work_injection = separate_fluxes(pressure_work_cell_flux)
    S_escape, S_injection = separate_fluxes(S_cell_flux)
    H_escape, H_injection = separate_fluxes(H_cell_flux)
    K_escape, K_injection = separate_fluxes(K_cell_flux)

    # save surface polydata to file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(out_surface_flux_fn)
    writer.SetInputConnection(vtk_polydata.GetOutputPort())
    writer.Write()

    # VOLUME
    # Calculate energies in the volume too

    # datafile reader
    f = pt.vlsvfile.VlsvReader(file_name=fn)
    # volume flags from SDF file
    f_SDF = pt.vlsvfile.VlsvReader(file_name=in_SDF_fn)
    region_SDF = f_SDF.read_variable("SDF")
    region_flags = np.where(region_SDF <= 0, 1, 0) # make region from SDF (assume that SDF is accurate)


    cellids = f.read_variable("CellID")
    region_cellids = cellids[region_flags==1]
    celldxs = f.get_cell_dx(region_cellids)
    dVs = np.prod(celldxs, axis=1) # volumes of cells inside volume

    nonregion_cellids = cellids[region_flags==0]
    nondVs = np.prod(f.get_cell_dx(nonregion_cellids), axis=1)


    # Energy densities
    P_th = f.read_variable("vg_pressure", cellids=region_cellids)
    P_dyn = f.read_variable("vg_pdyn", cellids=region_cellids)
    P_mag = f.read_variable("vg_p_magnetic", cellids=region_cellids)
    U_th = (P_th)/((5/3)-1)
    U_kin = (0.5)*(P_dyn) # vg_pdyn = rho_m*V^2
    U_mag = P_mag # vg_p_magnetic = B / 2.0 / mu_0
    U_tot = U_mag+U_th+U_kin


    # energies
    E_th = U_th*dVs
    E_kin = U_kin*dVs
    E_mag = U_mag*dVs
    E_tot = U_tot*dVs

    V_tot = np.sum(dVs) # total region volume

    Emag_tot = np.sum(E_mag)
    Eth_tot = np.sum(E_th)
    Ekin_tot = np.sum(E_kin)
    Total_Energy = Emag_tot+Eth_tot+Ekin_tot


    # also save energy fluxes (not through surface, just at all points)
    # not sure how meaningful but could be nice to see

    rhom = f.read_variable("vg_rhom")
    V_norm = f.read_variable("vg_v",  operator="magnitude")
    V = f.read_variable("vg_v")
    p_tensor = f.read_variable("vg_ptensor") #change
    p_diag = f.read_variable("vg_ptensor_diagonal")
    poynting = f.read_variable("vg_poynting")

    n_points = len(cellids)
    rhom = rhom.reshape(n_points, 1)
    V_norm = V_norm.reshape(n_points, 1)
    p_diag_sum = np.sum(p_diag, axis=1).reshape(n_points, 1)

    F_kinetic = (0.5*rhom*V_norm**2*V)
    F_thermal = (0.5 * p_diag_sum * V)
    F_pressure = (p_tensor @ V[:,:,None])[:,:,0]
    F_magnetic = poynting
    F_total = F_kinetic + F_thermal + F_pressure + F_magnetic


    # save j.E (Joule heating) and pressure strain over volume
    # check the E_vol?
    p_strain = f.read_variable("vg_p_strain", cellids=region_cellids)
    j = f.read_variable("vg_j", cellids=region_cellids)
    E = f.read_variable("vg_e_vol", cellids=region_cellids)
    j_dot_E = np.sum(j*E, axis=-1)
    j_dot_E_vol = np.sum(j_dot_E * dVs)
    p_strain_vol = np.sum(p_strain * dVs)


    if True: # save energies to vlsv grid

        # arrays for saving to vlsv so that cells outside region are nan
        U_thermal = np.full(len(cellids), np.nan)
        U_thermal[region_flags==1] = U_th

        U_kinetic = np.full(len(cellids), np.nan)
        U_kinetic[region_flags==1] = U_kin

        U_magnetic = np.full(len(cellids), np.nan)
        U_magnetic[region_flags==1] = U_mag

        U_total = np.full(len(cellids), np.nan)
        U_total[region_flags==1] = U_tot

        jE = np.full(len(cellids), np.nan)
        jE[region_flags==1] = j_dot_E

        pressure_strain = np.full(len(cellids), np.nan)
        pressure_strain[region_flags==1] = p_strain

        # write to vlsv file
        vlsv_writer = pt.vlsvfile.VlsvWriter(f, out_energy_vlsv_fn)
        vlsv_writer.copy_variables_list(f, ["CellID", "vg_rho"])
        vlsv_writer.copy_variables_list(f_SDF, ["SDF"])
        vlsv_writer.write_variable_info(pt.calculations.VariableInfo(U_thermal, "U_thermal", "J/m3", latex="", latexunits=""),"SpatialGrid",1,)
        vlsv_writer.write_variable_info(pt.calculations.VariableInfo(U_kinetic, "U_kinetic", "J/m3", latex="", latexunits=""),"SpatialGrid",1,)
        vlsv_writer.write_variable_info(pt.calculations.VariableInfo(U_magnetic, "U_magnetic", "J/m3", latex="", latexunits=""),"SpatialGrid",1,)
        vlsv_writer.write_variable_info(pt.calculations.VariableInfo(U_total, "U_total", "J/m3", latex="", latexunits=""),"SpatialGrid",1,)

        vlsv_writer.write_variable_info(pt.calculations.VariableInfo(F_thermal, "thermal_flux", "W/m2", latex="", latexunits=""),"SpatialGrid",1,)
        vlsv_writer.write_variable_info(pt.calculations.VariableInfo(F_kinetic, "kinetic_flux", "W/m2", latex="", latexunits=""),"SpatialGrid",1,)
        vlsv_writer.write_variable_info(pt.calculations.VariableInfo(F_pressure, "pressure_flux", "W/m2", latex="", latexunits=""),"SpatialGrid",1,)
        vlsv_writer.write_variable_info(pt.calculations.VariableInfo(F_magnetic, "poynting_flux", "W/m2", latex="", latexunits=""),"SpatialGrid",1,)
        vlsv_writer.write_variable_info(pt.calculations.VariableInfo(F_total, "total_flux", "W/m2", latex="", latexunits=""),"SpatialGrid",1,)

        vlsv_writer.write_variable_info(pt.calculations.VariableInfo(pressure_strain, "pressure_strain", "W/m3", latex="", latexunits=""),"SpatialGrid",1,)
        vlsv_writer.write_variable_info(pt.calculations.VariableInfo(jE, "j.E", "W/m3", latex="", latexunits=""),"SpatialGrid",1,)




    if True: #use pandas, easier option, could add manual csv write later
        import pandas as pd
        simtime = int(f.read_parameter("time"))
         # save *everything*
        data_dict = {"K through surface [J/s]":np.sum(K_cell_flux),#, axis=0),
             "S through surface [J/s]":np.sum(S_cell_flux),#, axis=0),
             "H through surface [J/s]":np.sum(H_cell_flux),#, axis=0),
             "kinetic flux through surface [J/s]":np.sum(kinetic_cell_flux),#, axis=0),
             "thermal flux through surface [J/s]":np.sum(thermal_cell_flux),#, axis=0),
             "pressure flux through surface [J/s]":np.sum(pressure_work_cell_flux), # this could be named better

             "Kinetic escape [J/s]": np.sum(kinetic_escape),
             "Kinetic injection [J/s]": np.sum(kinetic_injection),
             "Thermal escape [J/s]": np.sum(thermal_escape),
             "Thermal injection [J/s]": np.sum(thermal_injection),
             "Pressure escape [J/s]": np.sum(pressure_work_escape),
             "Pressure injection [J/s]": np.sum(pressure_work_injection),
             "Poynting escape [J/s]": np.sum(S_escape),
             "Poynting injection [J/s]": np.sum(S_injection),
             "Hydrodynamic escape [J/s]": np.sum(H_escape),
             "Hydrodynamic injection [J/s]": np.sum(H_injection),
             "Total escape [J/s]": np.sum(K_escape),
             "Total injection [J/s]": np.sum(K_injection),

             "U thermal [J]": Eth_tot,
             "U kinetic [J]": Ekin_tot,
             "U magnetic [J]": Emag_tot,
             "U total [J]": Total_Energy,

             "Pressure strain over volume [J/s]": p_strain_vol,
             "j.E over volume [J/s]": j_dot_E_vol,
             "Total region volume [m^3]": V_tot,
             "Total surface area [m^2]": np.sum(areas)
            }

        try: # existing file
            df = pd.read_csv(out_csv, index_col="time")
        except FileNotFoundError: # new file
            df = pd.DataFrame()
            df.index.name = "time"

        for col in data_dict:
            if col not in df.columns:
                df[col] = np.nan

        df.loc[simtime, data_dict.keys()] = data_dict.values()
        df.sort_index(inplace=True)

        df.to_csv(out_csv, index=True, float_format="%.17e")











def main():

    if len(sys.argv) == 2:
        timeid = int(sys.argv[1])
        areaname = "magnetopause"
    elif len(sys.argv) == 3:
        timeid = int(sys.argv[1])
        areaname = sys.argv[2]
    else:
        timeid = 1300
        areaname = "magnetopause"

    print("index:", str(timeid))

    datafilen = "/home/group/spacephysics/vlasiator/data/L0/3D/FHA/bulk1/bulk1.{:07d}.vlsv".format(timeid)

    in_path = "/turso/group/spacephysics/vlasiator/data/L1/3D/FHA/region_ids/"+areaname+"/"
    out_path = "FHAenergies/final/"+areaname+"/"


    SDF_vlsv = in_path+"FHA_{:07d}_".format(timeid)+areaname+"_SDF.vlsv"
    surface_vtp = in_path+"FHA_{:07d}_".format(timeid)+areaname+"_surface.vtp"

    surface_flux_vtp = out_path+"FHA_{:07d}_".format(timeid)+areaname+"_surface_flux.vtp"
    energy_volume_vlsv = out_path+"FHA_{:07d}_".format(timeid)+areaname+"_energy_density.vlsv"

    out_csv = out_path+"energies.csv"
    f_data = pt.vlsvfile.VlsvReader(file_name=datafilen)


    calculate_energies(datafilen, surface_vtp, surface_flux_vtp, SDF_vlsv, energy_volume_vlsv, out_path, out_csv)



if __name__ == "__main__":

    main()
