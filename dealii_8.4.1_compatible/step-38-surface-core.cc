/*------------------------------------------------------------------------
 * This program aims to simulate a surface-core structure, with different
 * growth rates in each part.
 *
 * Since this is a large-deformation problem, we need a calss of Quadrature 
 * Point History to store deformation history and associated stress.
 *
 * Bulk energy and surface energy are seperate.
 *
 * Reference: deal.II step-38 for surface problem.
 *                            1)how to create surface mesh.
 *                            2)surface derivatives(or tangential gradient) of a function.
 *            deal.II step-44 for 1) Newton-Raphson minimization of Energy.
 *                                2) quadrature poin history.
 *            Mechanics of Thin Film and Surface with Growth
 *
 * Le Yang
 * 2016
 *-------------------------------------------------------------------------*/



// @sect3{Include files}

// See step-4 and step-7 for the following include files.
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <fstream>
#include <iostream>


namespace surface_growth 
{
  using namespace dealii;

  template <int dim>
  class Material_Neo_Hook
  {
  public:
      Material_Neo_Hook(const double mu, donst double nu)
      {
      }

      ~Material_Neo_Hook()
      {}

      void update_material_data(const Tensor<2, dim> &F) {
      }

      SymmetricTensor<2, dim> get_tau(){
          return get_tau_iso() + get_tau_vol();
      }

      SymmetricTensor<2, dim> get_Jc(){
          return get_Jc_vol() + get_Jc_iso();
      }

      double get_dPsi_vol_dJ() const{
      }

      double get_d2Psi_vol_dJ2() const{
      }

  private:
      SymmetricTensor<2, dim> get_tau_iso(){
      }

      SymmetricTensor<2, dim> get_tau_vol(){
      }

      SymmetricTensor<2, dim> get_Jc_iso(){
          const SymmetricTensor<2, dim> tau_bar = get_tau_bar();

      }

      SymmetricTensor<2, dim> get_Jc_vol(){
      } 

  protected:
      const double c_1;
      double det_F;
      SymmetricTensor<2, dim> b_bar;
  };

  template <int dim>
  class Surface_Material
  {
  public:
      Surface_Material(const double mu, donst double nu)
      {
      }

      ~Surface_Material()
      {}

      void update_material_data(const Tensor<2, dim> &F) {
      }

      SymmetricTensor<2, dim> get_tau(){
          return get_tau_iso() + get_tau_vol();
      }

      SymmetricTensor<2, dim> get_Jc(){
          return get_Jc_vol() + get_Jc_iso();
      }

      double get_dPsi_vol_dJ() const{
      }

      double get_d2Psi_vol_dJ2() const{
      }

  private:
      SymmetricTensor<2, dim> get_tau_iso(){
      }

      SymmetricTensor<2, dim> get_tau_vol(){
      }

      SymmetricTensor<2, dim> get_Jc_iso(){
          const SymmetricTensor<2, dim> tau_bar = get_tau_bar();

      }

      SymmetricTensor<2, dim> get_Jc_vol(){
      } 

  protected:
      const double c_1;
      double det_F;
      SymmetricTensor<2, dim> b_bar;
  };

  template <int dim>
  class PointHistory
  {
  public:
      PointHistory():
          material(NULL),
          F_inv(StandardTensors<dim>::I),
          tau(SymmetricTensor<2, dim>()),
          d2Psi_vol_dJ2(0.0),
          dPsi_vol_dJ(0.0),
          Jc(SymmetricTensor<4, dim>())
      {}

      virtual ~PointHistory(){
          delete material;
          material = NULL;
      }

      void setup_lqp (){
          if (dim == 3)material = new Material_Neo_Hook<dim>(mu, nu);
          if (dim == 2)material = new Surface_material<dim>(mu, nu);
          update_values(Tensor<2, dim>() );
      }

      void update_values(const Tensor<2, dim> &Grad_u_n){
          const Tensor<2, dim> F = Tensor<2, dim>(StandardTensors<dim>::I + Grad_u_n;
          material->update_material_data(F);
          F_inv = invert(F);
          tau = material->get_tau();
          Jc = material->get_Jc();
          dPsi_vol_dJ = material->get_dPsi_vol_dJ();
          d2Psi_vol_dJ2 = material->get_d2Psi_vol_dJ2();
      }

  private:
      Material_New_Hook<dim> *material;
      Tensor<2, dim> F_inv;
      SymmetricTensor<2, dim> tau;
      double d2Psi_vol_dJ2;
      double dPsi_vol_dJ;
      SymmetricTensor<4, dim> Jc;
  };


  // @sect3{The <code>SurfaceCoreProblem</code> class template}

  // The essential differences are these:
  //
  // - The template parameter now denotes the dimensionality of the embedding
  //   space, which is no longer the same as the dimensionality of the domain
  //   and the triangulation on which we compute. We indicate this by calling
  //   the parameter @p spacedim , and introducing a constant @p dim equal to
  //   the dimensionality of the domain -- here equal to
  //   <code>spacedim-1</code>.
  // - All member variables that have geometric aspects now need to know about
  //   both their own dimensionality as well as that of the embedding
  //   space. Consequently, we need to specify both of their template
  //   parameters one for the dimension of the mesh @p dim, and the other for
  //   the dimension of the embedding space, @p spacedim. 
  //   See step-34 for a deeper explanation.
  // - We need an object that describes which kind of mapping to use from the
  //   reference cell to the cells that the triangulation is composed of. The
  //   classes derived from the Mapping base class do exactly this. Throughout
  //   most of deal.II, if you don't do anything at all, the library assumes
  //   that you want an object of kind MappingQ1 that uses a (bi-, tri-)linear
  //   mapping. In many cases, this is quite sufficient, which is why the use
  //   of these objects is mostly optional: for example, if you have a
  //   polygonal two-dimensional domain in two-dimensional space, a bilinear
  //   mapping of the reference cell to the cells of the triangulation yields
  //   an exact representation of the domain. If you have a curved domain, one
  //   may want to use a higher order mapping for those cells that lie at the
  //   boundary of the domain -- this is what we did in step-11, for
  //   example. However, here we have a curved domain, not just a curved
  //   boundary, and while we can approximate it with bilinearly mapped cells,
  //   it is really only prudent to use a higher order mapping for all
  //   cells. Consequently, this class has a member variable of type MappingQ;
  //   we will choose the polynomial degree of the mapping equal to the
  //   polynomial degree of the finite element used in the computations to
  //   ensure optimal approximation, though this iso-parametricity is not
  //   required.
  template <int spacedim>
  class SurfaceCoreProblem
  {
  public:
    SurfaceCoreProblem (const unsigned degree = 2);
    void run ();

  private:
    static const unsigned int dim = spacedim-1;

    void make_grid_and_dofs ();
    void assemble_system ();
    void solve ();
    void output_results () const;
    void compute_error () const;


    Triangulation<spacedim>       triangulation;//for bulk
    std::vector<PointHistory<spacedim> > qph;
    std::vector<PointHistory<dim> > qph_s;
    FE_Q<spacedim>                fe;
    DofHandler<spacedim>          dof_handler;
    MappingQ<dim, spacedim>       mapping;

    SparsityPattern               sparsity_pattern;
    SparseMatrix<double>          system_matrix;

    Vector<double>                solution;
    Vector<double>                system_rhs;
  };


  // @sect3{Equation data}

  // Next, let us define the classes that describe the
  // right hand sides of the problem. This is in analogy to step-4 and step-7
  // where we also defined such objects. Given the discussion in the
  // introduction, the actual formulas should be self-explanatory. A point of
  // interest may be how we define the value and gradient functions for the 2d
  // and 3d cases separately, using explicit specializations of the general
  // template. An alternative to doing it this way might have been to define
  // the general template and have a <code>switch</code> statement (or a
  // sequence of <code>if</code>s) for each possible value of the spatial
  // dimension.


  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide () : Function<dim>() {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
  };

  template <>
  double
  RightHandSide<2>::value (const Point<2> &p,
                           const unsigned int /*component*/) const
  {
    return ( -8. * p(0) * p(1) );
  }


  template <>
  double
  RightHandSide<3>::value (const Point<3> &p,
                           const unsigned int /*component*/) const
  {
    using numbers::PI;

    Tensor<2,3> hessian;

    hessian[0][0] = -PI*PI*sin(PI*p(0))*cos(PI*p(1))*exp(p(2));
    hessian[1][1] = -PI*PI*sin(PI*p(0))*cos(PI*p(1))*exp(p(2));
    hessian[2][2] = sin(PI*p(0))*cos(PI*p(1))*exp(p(2));

    hessian[0][1] = -PI*PI*cos(PI*p(0))*sin(PI*p(1))*exp(p(2));
    hessian[1][0] = -PI*PI*cos(PI*p(0))*sin(PI*p(1))*exp(p(2));

    hessian[0][2] = PI*cos(PI*p(0))*cos(PI*p(1))*exp(p(2));
    hessian[2][0] = PI*cos(PI*p(0))*cos(PI*p(1))*exp(p(2));

    hessian[1][2] = -PI*sin(PI*p(0))*sin(PI*p(1))*exp(p(2));
    hessian[2][1] = -PI*sin(PI*p(0))*sin(PI*p(1))*exp(p(2));

    Tensor<1,3> gradient;
    gradient[0] = PI * cos(PI*p(0))*cos(PI*p(1))*exp(p(2));
    gradient[1] = - PI * sin(PI*p(0))*sin(PI*p(1))*exp(p(2));
    gradient[2] = sin(PI*p(0))*cos(PI*p(1))*exp(p(2));

    Point<3> normal = p;
    normal /= p.norm();

    return (- trace(hessian)
            + 2 * (gradient * normal)
            + (hessian * normal) * normal);
  }


  // @sect3{Implementation of the <code>SurfaceCoreProblem</code> class}

  // The rest of the program is actually quite unspectacular if you know
  // step-4. Our first step is to define the constructor, setting the
  // polynomial degree of the finite element and mapping, and associating the
  // DoF handler to the triangulation:
  template <int spacedim>
  SurfaceCoreProblem<spacedim>::
  SurfaceCoreProblem (const unsigned degree)
    :
    fe (degree),
    dof_handler(triangulation),
    mapping (degree)
  {}

  template <int spacedim>
  SurfaceCoreProblem<spacedim>::setup_qph()
  {
    std::cout << "    Setting up quadrature point data..." << std::endl;

    {
      triangulation.clear_user_data();
      {
          std::vector<PointHistory<spacedim> > tmp;
          tmp.swap(qph);
          std::vector<PointHistory<dim> > tmp2;
          tmp2.swap(qph_s);
      }
      qph.resize(triangulation.n_active_cells() * n_q_points);
      unsigned int num_faces_on_boundary0 = 0;
      unsigned int history_index = 0;
      for (typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active();
              cell != triangulation.end(); ++ cell){
          cell->set_user_pointer(&qph[history_index]);
          history_index += n_q_points;
          //count num of faces on boundary 0
          for(unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face){
              if (cell->face(face)->at_boundary() == true && cell->face(face)->boundary_id() == 0){
                  num_faces_on_boundary0 ++;
              }
          }
      }
      
      qph_s.resize(num_faces_on_boundary0 * n_f_q_points);
      unsigned int history_index_s = 0;
      for (typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active();
              cell != triangulation.end(); ++ cell){
          for(unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face){
              if (cell->face(face)->at_boundary() == true && cell->face(face)->boundary_id() == 0){
                  //check usage for set_user_pointer(...)
                  //add if necessary
                  cell->face(face)->set_user_pointer(&qph_s[history_index_s]);
                  history_index_s += n_q_points_f;
              }
          }
      }    

      Assert(history_index == qph.size(),
             ExcInternalError());
      
    }
    
    for (typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active();
                         cell != triangulation.end(); ++cell){
        PointHistory<dim> *lqph = reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            lqph[q_point].setup_lqp(parameters);
    }
  }

  // @sect4{SurfaceCoreProblem::make_grid_and_dofs}

  // The next step is to create the mesh, distribute degrees of freedom, and
  // set up the various variables that describe the linear system. All of
  // these steps are standard with the exception of how to create a mesh that
  // describes a surface. We could generate a mesh for the domain we are
  // interested in, generate a triangulation using a mesh generator, and read
  // it in using the GridIn class. Or, as we do here, we generate the mesh
  // using the facilities in the GridGenerator namespace.
  //
  // In particular, what we're going to do is this (enclosed between the set
  // of braces below): we generate a <code>spacedim</code> dimensional mesh
  // for the half disk (in 2d) or half ball (in 3d), using the
  // GridGenerator::half_hyper_ball function. This function sets the boundary
  // indicators of all faces on the outside of the boundary to zero for the
  // ones located on the perimeter of the disk/ball, and one on the straight
  // part that splits the full disk/ball into two halves. The next step is the
  // main point: The GridGenerator::extract_boundary_mesh function creates a mesh
  // that consists of those cells that are the faces of the previous mesh,
  // i.e. it describes the <i>surface</i> cells of the original (volume)
  // mesh. However, we do not want all faces: only those on the perimeter of
  // the disk or ball which carry boundary indicator zero; we can select these
  // cells using a set of boundary indicators that we pass to
  // GridGenerator::extract_boundary_mesh.
  //
  // There is one point that needs to be mentioned. In order to refine a
  // surface mesh appropriately if the manifold is curved (similarly to
  // refining the faces of cells that are adjacent to a curved boundary), the
  // triangulation has to have an object attached to it that describes where
  // new vertices should be located. If you don't attach such a boundary
  // object, they will be located halfway between existing vertices; this is
  // appropriate if you have a domain with straight boundaries (e.g. a
  // polygon) but not when, as here, the manifold has curvature. So for things
  // to work properly, we need to attach a manifold object to our (surface)
  // triangulation, in much the same way as we've already done in 1d for the
  // boundary. We create such an object (with indefinite, <code>static</code>,
  // lifetime) at the top of the function and attach it to the triangulation
  // for all cells with boundary indicator zero that will be created
  // henceforth.
  //
  // The final step in creating the mesh is to refine it a number of
  // times. The rest of the function is the same as in previous tutorial
  // programs.
  template <int spacedim>
  void SurfaceCoreProblem<spacedim>::make_grid_and_dofs ()
  {
    static SphericalManifold<dim,spacedim> surface_description;

    {
      Triangulation<spacedim> triangulation;
      GridGenerator::half_hyper_ball(triangulation);
    }
    triangulation_s.set_all_manifold_ids(0);
    triangulation_s.set_manifold (0, surface_description);

    triangulation_s.refine_global(4);

    std::cout << "Mesh has " << triangulation.n_active_cells()
              << " cells."
              << std::endl;

    dof_handler.distribute_dofs (fe);

    std::cout << "Mesh has " << dof_handler.n_dofs()
              << " degrees of freedom." << std::endl;

    DynamicSparsityPattern dsp (dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern (dof_handler, dsp);
    sparsity_pattern.copy_from (dsp);

    system_matrix.reinit (sparsity_pattern);

    solution.reinit (dof_handler.n_dofs());
    system_rhs.reinit (dof_handler.n_dofs());
  }


  // @sect4{SurfaceCoreProblem::assemble_system}

  // The following is the central function of this program, assembling the
  // matrix that corresponds to the surface Laplacian (Laplace-Beltrami
  // operator). Maybe surprisingly, it actually looks exactly the same as for
  // the regular Laplace operator discussed in, for example, step-4. The key
  // is that the FEValues::shape_gradient function does the magic: It returns
  // the surface gradient $\nabla_K \phi_i(x_q)$ of the $i$th shape function
  // at the $q$th quadrature point. The rest then does not need any changes
  // either:
  template <int spacedim>
  void SurfaceCoreProblem<spacedim>::assemble_system ()
  {
    system_matrix = 0;
    system_rhs = 0;

    const QGauss<dim>  quadrature_formula(2*fe.degree);
    FEValues<dim,spacedim> fe_values (mapping, fe, quadrature_formula,
                                      update_values              |
                                      update_gradients           |
                                      update_quadrature_points   |
                                      update_JxW_values);

    const unsigned int        dofs_per_cell = fe.dofs_per_cell;
    const unsigned int        n_q_points    = quadrature_formula.size();

    FullMatrix<double>        cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>            cell_rhs (dofs_per_cell);

    std::vector<double>       rhs_values(n_q_points);
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    const RightHandSide<spacedim> rhs;

    for (typename DoFHandler<dim,spacedim>::active_cell_iterator
         cell = dof_handler.begin_active(),
         endc = dof_handler.end();
         cell!=endc; ++cell)
      {
        cell_matrix = 0;
        cell_rhs = 0;
        
        PointHistory<spacedim> *lqph = reinterpret_cast<PointHistory<dim>* >(cell->user_pointer());
        //*********************************************************************************
        //**************************bulk part**********************************************
        //*********************************************************************************
        fe_values.reinit (cell);

        rhs.value_list (fe_values.get_quadrature_points(), rhs_values);

        //*****************initialize shape_value and shape_grad*************************** 
        std::vector<std::vector<double> >                      Nx(n_q_points,
                                                                  std::vector<double>(fe_cell.dofs_per_cell));
        std::vector<std::vector<Tensor<2, spacedim> > >             grad_Nx(n_q_points,
                                                                  std::vector<Tensor<2, spacedim> >(fe_cell.dofs_per_cell));
        std::vector<std::vector<SymmetricTensor<2, spacedim> > >    symm_grad_Nx(n_q_points,
                                                                  std::vector<SymmetricTensor<2, spacedim> >(fe_cell.dofs_per_cell));

        for (unsigned int q_point=0; q_point < n_q_points; ++ q_point){
            const Tensor<2, spacedim> F_inv = lqph[q_point].get_F_inv();
            for (unsigned in k=0; k < dofs_per_cell; ++k){
                grad_Nx[q_point][k] = fe_values.gradient(k, q_point) * F_inv;
                symm_grad_Nx[q_point][k] = symmetrize(grad_Nx[q_point][k]); 
        }
        
        //******************assembly stiffness matrix**************************************
        for (unsigned int q_point=0; q_point < n_q_points; ++ q_point){
            const Tensor<2, spacedim> tau         = lqph[q_point].get_tau();
            const SymmetricTensor<4, spacedim> Jc = lqph[q_point].get_Jc();
            const double d2Psi_vol_dJ2       = lqph[q_point].get_d2Psi_vol_dJ2();
            const double det_F               = lqph[q_point].get_det_F();
            for (unsigned int i=0; i<dofs_per_cell; ++i){
                const unsigned int component_i = fe.system_to_component_index(i).first;
                for (unsigned int j=0; j<dofs_per_cell; ++j){
                    const unsigned int component_j = fe.system_to_component_index(j).first;
                    cell_matrix(i,j) += symm_grad_Nx[q_point][i] * Jc
                                        * symm_grad_Nx[q_point][j] * fe_values.JxW(q_point);
                    if (component_i == component_j)
                        cell_matrix(i,j) += grad_Nx[q_point][i][component_i] * tau *
                                      grad_Nx[q_point][j][component_j] *
                                      fe_values.JxW(q_point);
                //surface part is added later 
                }
            }
        }

        //*******************assemble rhs**************************************************
        for (unsigned int q_point=0; q_point<n_q_points; ++q_point){
            const SymmetricTensor<2, spacedim> tau = lqph[q_point].get_tau();
            const double det_F = lqph[q_point].get_det_F();
            const double dPsi_vol_dJ = lqph[q_point].get_dPsi_vol_dJ();

            for (unsigned int i=0; i<dofs_per_cell; ++i){
                cell_rhs(i) += symm_grad_Nx[q_point][i] * tau * fe_values.JxW(q_point);
            }
            //surface part is added later
        }
        
        //***************************************************************************
        //**********************surface part*****************************************
        //***************************************************************************
        //Attention!!! Need to find relation between dof_on_face and dof_on_cell
        //e.g. face_to_cell_dof(i)
        std::vector<std::vector<double> >                      Nx_f(n_q_points,
                                                                  std::vector<double>(fe_cell_face.dofs_per_cell_face));
        std::vector<std::vector<Tensor<2, dim> > >             grad_Nx_f(n_q_points,
                                                                  std::vector<Tensor<2, dim> >(fe_cell_face.dofs_per_cell_face));
        std::vector<std::vector<SymmetricTensor<2, dim> > >    symm_grad_Nx_f(n_q_points,
                                                                  std::vector<SymmetricTensor<2, dim> >(fe_cell_face.dofs_per_cell_face));

        
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face){
            if (cell->face(face)->at_boundary() == true && cell->face(face)->boundary_id() == 0){
                //check usage for use_pointer()
                PointHistory<dim> *lqph_s = reinterpret_cast<PointHistory<dim>* >(cell->face(face)->user_pointer());
                fe_face_values.reinit(cell, face);
                //**************initialize shape_value and shape_grad*************************
                for (unsigned int f_q_point=0; f_q_point < n_q_points_f; ++f_q_point){
                    const Tensor<2, dim> F_inv_f = lqph[f_q_point].get_F_inv();
                    for (unsigned in k=0; k < dofs_per_cell_face; ++k){
                        grad_Nx_f[f_q_point][k] = fe_face_values.gradient(k, f_q_point) * F_inv;
                        symm_grad_Nx_f[f_q_point][k] = symmetrize(grad_Nx_f[f_q_point][k]); 
                }
                
                for (unsigned int f_q_point = 0; f_q_point < n_q_points_f; ++f_q_point){
                    //uncomment to calculate external pressure applied to this boundary
                    //const Tensor<1, dim> &N = fe_face_values.normal_vector(f_q_point);
                    //const Tensor<1, dim> traction = pressure * N;
                    //const Tensor<1, dim> tau_surface = tau - tau * N;
                    //calculate internal stress of curved surface
                    const Tensor<2, dim> tau_s         = lqph_s[f_q_point].get_tau();
                    const SymmetricTensor<4, dim> Jc = lqph_s[f_q_point].get_Jc();
                    const double d2Psi_vol_dJ2       = lqph_s[f_q_point].get_d2Psi_vol_dJ2();
                    const double det_F               = lqph_s[f_q_point].get_det_F();
                    //We need to convert i-th and j-th(dofs_per_cell_face) to dofs_per_cell 
                    for (unsigned int i = 0; i < dofs_per_cell_face; ++i){
                        const unsigned int component_i = fe.system_to_component_index(i).first;
                        for (unsigned int j = 0; j < dofs_per_cell_face; ++j){
                            const unsigned int component_j = fe.system_to_comonent_index(j).first;
                            cell_matrix(face_to_cell_dof(i),face_to_cell_dof(j)) += grad_Nx_f[f_q_point][i] * Jc
                                                * grad_Nx_f[f_q_point][j] * fe_face_values.JxW(f_q_point);
                            if (component_i == component_j)
                                cell_matrix(face_to_cell_dof(i),face_to_cell_dof(j)) += grad_Nx_f[f_q_point][i][component_i] * tau_s *
                                                    grad_Nx_f[f_q_point][j][component_j] * fe_face_values.JxW(f_q_point);
                        }
                    } 
                    
                    for (unsigned int i = 0; i < dofs_per_cell; ++i){
                        const unsigned int component_i = fe.system_to_component_index(i).first;
                        cell_rhs(face_to_cell_dof(i)) += grad_Nx_f[i] * tau_surface * fe_face_values.JxW(f_q_point);
                    } 
                }
            }
        }                     
                    
        cell->get_dof_indices (local_dof_indices);
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
            for (unsigned int j=0; j<dofs_per_cell; ++j)
              system_matrix.add (local_dof_indices[i],
                                 local_dof_indices[j],
                                 cell_matrix(i,j));

            system_rhs(local_dof_indices[i]) += cell_rhs(i);
          }
      }

    std::map<types::global_dof_index,double> boundary_values;
    VectorTools::interpolate_boundary_values (mapping,
                                              dof_handler,
                                              0,
                                              Solution<spacedim>(),
                                              boundary_values);

    MatrixTools::apply_boundary_values (boundary_values,
                                        system_matrix,
                                        solution,
                                        system_rhs,false);
  }



  // @sect4{SurfaceCoreProblem::solve}

  // The next function is the one that solves the linear system. Here, too, no
  // changes are necessary:
  template <int spacedim>
  void SurfaceCoreProblem<spacedim>::solve ()
  {
    SolverControl solver_control (solution.size(),
                                  1e-7 * system_rhs.l2_norm());
    SolverCG<>    cg (solver_control);

    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);

    cg.solve (system_matrix, solution, system_rhs,
              preconditioner);
  }



  // @sect4{SurfaceCoreProblem::output_result}

  // This is the function that generates graphical output from the
  // solution. Most of it is boilerplate code, but there are two points worth
  // pointing out:
  //
  // - The DataOut::add_data_vector function can take two kinds of vectors:
  //   Either vectors that have one value per degree of freedom defined by the
  //   DoFHandler object previously attached via DataOut::attach_dof_handler;
  //   and vectors that have one value for each cell of the triangulation, for
  //   example to output estimated errors for each cell. Typically, the
  //   DataOut class knows to tell these two kinds of vectors apart: there are
  //   almost always more degrees of freedom than cells, so we can
  //   differentiate by the two kinds looking at the length of a vector. We
  //   could do the same here, but only because we got lucky: we use a half
  //   sphere. If we had used the whole sphere as domain and $Q_1$ elements,
  //   we would have the same number of cells as vertices and consequently the
  //   two kinds of vectors would have the same number of elements. To avoid
  //   the resulting confusion, we have to tell the DataOut::add_data_vector
  //   function which kind of vector we have: DoF data. This is what the third
  //   argument to the function does.
  // - The DataOut::build_patches function can generate output that subdivides
  //   each cell so that visualization programs can resolve curved manifolds
  //   or higher polynomial degree shape functions better. We here subdivide
  //   each element in each coordinate direction as many times as the
  //   polynomial degree of the finite element in use.
  template <int spacedim>
  void SurfaceCoreProblem<spacedim>::output_results () const
  {
    DataOut<dim,DoFHandler<dim,spacedim> > data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (solution,
                              "solution",
                              DataOut<dim,DoFHandler<dim,spacedim> >::type_dof_data);
    data_out.build_patches (mapping,
                            mapping.get_degree());

    std::string filename ("solution-");
    filename += static_cast<char>('0'+spacedim);
    filename += "d.vtk";
    std::ofstream output (filename.c_str());
    data_out.write_vtk (output);
  }



  // @sect4{SurfaceCoreProblem::compute_error}

  // This is the last piece of functionality: we want to compute the error in
  // the numerical solution. It is a verbatim copy of the code previously
  // shown and discussed in step-7. As mentioned in the introduction, the
  // <code>Solution</code> class provides the (tangential) gradient of the
  // solution. To avoid evaluating the error only a superconvergence points,
  // we choose a quadrature rule of sufficiently high order.
  template <int spacedim>
  void SurfaceCoreProblem<spacedim>::compute_error () const
  {
    Vector<float> difference_per_cell (triangulation.n_active_cells());
    VectorTools::integrate_difference (mapping, dof_handler, solution,
                                       Solution<spacedim>(),
                                       difference_per_cell,
                                       QGauss<dim>(2*fe.degree+1),
                                       VectorTools::H1_norm);

    std::cout << "H1 error = "
              << difference_per_cell.l2_norm()
              << std::endl;
  }



  // @sect4{SurfaceCoreProblem::run}

  // The last function provides the top-level logic. Its contents are
  // self-explanatory:
  template <int spacedim>
  void SurfaceCoreProblem<spacedim>::run ()
  {
    make_grid_and_dofs();
    assemble_system ();
    solve ();
    output_results ();
    compute_error ();
  }
}


// @sect3{The main() function}

// The remainder of the program is taken up by the <code>main()</code>
// function. It follows exactly the general layout first introduced in step-6
// and used in all following tutorial programs:
int main ()
{
  try
    {
      using namespace dealii;
      using namespace Step38;

      SurfaceCoreProblem<3> laplace_beltrami;
      laplace_beltrami.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
