# l2_norm = np.linalg.norm(concentration_solution)
#     gradient_norm_squared = 0.0
#     for i in range(len(triangles)):
#         triangle_concentration = concentration_solution[triangles[i]]
#         gradient = compute_gradient(triangle_vertices[i], triangle_concentration)
#         gradient_norm_squared += np.dot(gradient, gradient)

#     l2_gradient_norm = np.sqrt(gradient_norm_squared)
#     print(f"Grad : {l2_gradient_norm}")

#     print(f"L2 : {l2_norm}")




# X_concentration = triangulation_results["vertices"][:, 0]
#     Y_concentration = triangulation_results["vertices"][:, 1]
#     Z_concentration = concentration_solution

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection="3d")
#     surf = ax.plot_trisurf(
#         X_concentration,
#         Y_concentration,
#         Z_concentration,
#         cmap="Grays",
#         edgecolor="none",
#     )
#     fig.colorbar(surf, shrink=0.5, aspect=5)
#     ax.set_title("Concentration")
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("u(x, y)")
#     plt.show()


    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # surf = ax.plot_trisurf(
    #     X_concentration,
    #     Y_concentration,
    #     Z_concentration,
    #     cmap="Grays",
    #     edgecolor="none",
    # )
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # ax.set_title("Concentration")
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("u(x, y)")
    # plt.show()
    
    
    
    #     l2_norm = np.linalg.norm(concentration_solution)
    # gradient_norm_squared = 0.0
    # for i in range(len(triangles)):
    #     triangle_concentration = concentration_solution[triangles[i]]
    #     gradient = compute_gradient(triangle_vertices[i], triangle_concentration)
    #     gradient_norm_squared += np.dot(gradient, gradient)

    # l2_gradient_norm = np.sqrt(gradient_norm_squared)
    # print(f"Grad : {l2_gradient_norm}")

    # print(f"L2 : {l2_norm}")
    
    
    
    # print(pressure_solution)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # surf = ax.plot_trisurf(
    #     X_concentration,
    #     Y_concentration,
    #     Z_concentration,
    #     cmap="Grays",
    #     edgecolor="none",
    # )
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # ax.set_title("Pressure")
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("u(x, y)")
    # plt.show()