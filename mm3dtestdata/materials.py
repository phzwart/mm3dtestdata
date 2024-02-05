import torch
import numpy as np

materials_data = [
{"Material": "Quartz",
 "Density (g/cm)": 2.65,
 "Electron Density (electrons/nm3)": 0.797,
 "Ca (atoms/nm3)": 0.0,
 " O (atoms/nm3)": 88.2,
 "Si (atoms/nm3)": 44.1,
 "Al (atoms/nm3)": 0.0,
 " K (atoms/nm3)": 0.0,
 "Fe (atoms/nm3)": 0.0},

{"Material": "Feldspar",
 "Density (g/cm)": 2.56,
 "Electron Density (electrons/nm3)": 0.637,
 "Ca (atoms/nm3)": 0.0,
 " O (atoms/nm3)": 73.6,
 "Si (atoms/nm3)": 27.6,
 "Al (atoms/nm3)": 9.2,
 " K (atoms/nm3)": 9.2,
 "Fe (atoms/nm3)": 0.0},

{"Material": "Calcite",
 "Density (g/cm)": 2.71,
 "Electron Density (electrons/nm3)": 0.815,
 "Ca (atoms/nm3)": 27.1,
 " O (atoms/nm3)": 81.2,
 "Si (atoms/nm3)": 0.0,
 "Al (atoms/nm3)": 0.0,
 " K (atoms/nm3)": 0.0,
 "Fe (atoms/nm3)": 0.0},

{"Material": "Iron Oxide",
 "Density (g/cm)": 5.24,
 "Electron Density (electrons/nm3)": 1.383,
 "Ca (atoms/nm3)": 0.0,
 " O (atoms/nm3)": 98.4,
 "Si (atoms/nm3)": 0.0,
 "Al (atoms/nm3)": 0.0,
 " K (atoms/nm3)": 0.0,
 "Fe (atoms/nm3)": 65.6},

{"Material": "Vaterite",
 "Density (g/cm)": 2.4,
 "Electron Density (electrons/nm3)": 0.722,
 "Ca (atoms/nm3)": 24.0,
 " O (atoms/nm3)": 71.9,
 "Si (atoms/nm3)": 0.0,
 "Al (atoms/nm3)": 0.0,
 " K (atoms/nm3)": 0.0,
 "Fe (atoms/nm3)": 0.0},

{"Material": "Calcite/Vaterite Mix",
 "Density (g/cm)": 0.0,
 "Electron Density (electrons/nm3)": 0.797,
 "Ca (atoms/nm3)": 26.46,
 " O (atoms/nm3)": 79.39,
 "Si (atoms/nm3)": 0.0,
 "Al (atoms/nm3)": 0.0,
 " K (atoms/nm3)": 0.0,
 "Fe (atoms/nm3)": 0.0},

{"Material": "Epon 812 Epoxy",
 "Density (g/cm)": 1.2,
 "Electron Density (electrons/nm3)": 0.386,
 "Ca (atoms/nm3)": 0.0,
 " O (atoms/nm3)": 14.1,
 "Si (atoms/nm3)": 0.0,
 "Al (atoms/nm3)": 0.0,
 " K (atoms/nm3)": 0.0,
 "Fe (atoms/nm3)": 0.0},

{"Material": "Vacuum",
 "Density (g/cm)": 0.0,
 "Electron Density (electrons/nm3)": 0.0,
 "Ca (atoms/nm3)": 0.0,
 " O (atoms/nm3)": 0.0,
 "Si (atoms/nm3)": 0.0,
 "Al (atoms/nm3)": 0.0,
 " K (atoms/nm3)": 0.0,
 "Fe (atoms/nm3)": 0.0}
]



composite_materials = [
    {"Name": "VEQF",
     0 : "Vacuum",
     1 : "Epon 812 Epoxy",
     2 : "Quartz",
     3 : "Feldspar",
     "Comments" : "Minimal contrast between Quartz (0.797) and Feldspar (0.637)"
    },

    {"Name": "VEQI",
     0 : "Vacuum",
     1 : "Epon 812 Epoxy",
     2 : "Quartz",
     3 : "Iron Oxide",
     "Comments" : "Large contrast between Quartz (0.797) and Iron Oxide (1.383)"
    },


    {"Name": "VEQM",
     0 : "Vacuum",
     1 : "Epon 812 Epoxy",
     2 : "Quartz",
     3 : "Calcite/Vaterite Mix",
     "Comments" : "No contrast between Quartz (0.797) and Calcite/Vaterite (0.797)"
    },
]

def find_material_by_name(name, materials=materials_data):
    """
    Finds a material by its name from a list of materials.

    Parameters:
    - name (str): The name of the material to find.
    - materials (list): The list of materials to search through.

    Returns:
    - dict: The dictionary of the material if found, otherwise None.
    """
    for material in materials:
        if material["Material"] == name:
            return material
    return None  # Return None if the material is not found


def build_composite_material_actions_XCT_SEM_EDX(composite_name,
                             elements=["Si",
                                       "Al",
                                       "Fe",
                                       "Ca",
                                       "Al",
                                       " K"]):
    """
    Builds arrays representing the electron density and elemental composition
    for a given composite material, simulating combined XCT and SEM-EDX analysis.

    Parameters:
    - composite_name (str): The name of the composite material.
    - elements (list): The list of elements to consider for spectral densities.

    Returns:
    - tuple: A tuple containing two numpy arrays:
        - The first array represents the electron density of each component material.
        - The second array represents the elemental composition of each component material.
    """
    all_ok = False
    object = None
    for item in composite_materials:
        if composite_name == item["Name"]:
            object = item
            all_ok = True
    if not all_ok:
        print("composite material %s not found"%composite_name)
    assert all_ok

    # we first build the tomographic densities.
    class_action_tomo = []
    for item in object:
        if type(item) is int:
            this_material = find_material_by_name(object[item])
            t = np.array([this_material["Electron Density (electrons/nm3)"]])
            class_action_tomo.append(t)
    class_action_tomo = np.column_stack(class_action_tomo)

    # now we need to build the spectral densities
    element_class_action = []
    for item in object:
        element_counter = np.zeros(len(elements))
        if type(item) is int:
            this_material = find_material_by_name(object[item])
            for item in this_material:
                if item[0:2] in elements:
                    indx = elements.index(item[0:2])
                    element_counter[indx] += this_material[item]
            element_class_action.append(element_counter)
    element_class_action = np.column_stack(element_class_action)
    print(element_class_action)
    return class_action_tomo, element_class_action






if __name__ == "__main__":
    print(build_composite_material_actions_XCT_SEM_EDX("VEQF", ["Si", "Al", " K"]))








