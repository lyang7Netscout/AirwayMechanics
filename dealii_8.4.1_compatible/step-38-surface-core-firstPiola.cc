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
 *                            2)first fundamental form of a surface.
 *                            3)surface derivatives(or tangential gradient) of a function.
 *            deal.II step-44 for 1) Newton-Raphson minimization of Energy.
 *                                2) quadrature poin history.
 *                                3) usage of FEValues and FEFaceValues together.
 *            One the mechanics of thin films and growing surfaces(2013) Alain Goriely, Ellen Kuhl
 *
 * Le Yang
 * 2016
 *-------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------
 * This version uses first Piola-Kirchhoff stress P
 * and elasticity tensor A
 * bulk and surface part are according to paper
 * bulk part is partially validated to be consistent with step-44
 *--------------------------------------------------------------------------*/

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
  
  namespace parameters{
      double mu = 0.5;
      double lambda = 0.3;
      double mu_s = 0.5;
      double lambda_s = 0.3;
  }

// @sect3{Some standard tensors}
// Now we define some frequently used second and fourth-order tensors:
  template <int dim>
  class StandardTensors
  {
  public:

    static const SymmetricTensor<2, dim> I;
    static const SymmetricTensor<4, dim> IxI;
    // $\mathcal{S}$, note that as we only use this fourth-order unit tensor
    // to operate on symmetric second-order tensors.  To maintain notation
    // consistent with Holzapfel (2001) we name the tensor $\mathcal{I}$
    static const SymmetricTensor<4, dim> II;
	// Fourth-order deviatoric tensor such that
	// dev{ . } = { . } - 1/{dim} * [ { . } :I]\I
    static const SymmetricTensor<4, dim> dev_P;
  };

  template <int dim>
  const SymmetricTensor<2, dim>
  StandardTensors<dim>::I = unit_symmetric_tensor<dim>();

  template <int dim>
  const SymmetricTensor<4, dim>
  StandardTensors<dim>::IxI = outer_product(I, I);

  template <int dim>
  const SymmetricTensor<4, dim>
  StandardTensors<dim>::II = identity_tensor<dim>();

  template <int dim>
  const SymmetricTensor<4, dim>
  StandardTensors<dim>::dev_P = deviator_tensor<dim>();

// @sect3{Materials for bulk and surface}
//
//
//
  template <int dim>
  class Material
  {
  public:
      Material(const double mu_, const double lambda_)
        :c_1(mu_ / 2.0),
         mu(mu_),
         lambda(lambda_),
         det_F(1.0)
      {
      }

      ~Material()
      {}

      void update_material_data(const Tensor<2, dim> &F_)=0;

      SymmetricTensor<2, dim> get_P()=0;

      SymmetricTensor<4, dim> get_A()=0; //fourth-order elasticity tensor in the reference setting

      double get_d2Psi_vol_dJ2() const{ return 0.0; }

  private:
      Tensor<2, dim> get_P_iso()=0;

      Tensor<2, dim> get_P_vol()=0;

      SymmetricTensor<4, dim> get_A_iso()=0;

      SymmetricTensor<4, dim> get_A_vol()=0;

  protected:
      const double c_1;
      const double mu;
      const double lambda;
      double det_F;
      Tensor<2, dim> F;
      Tensor<2, dim> F_inv;
  };

  template <int dim>
  class Material_Neo_Hook : public Material<dim>
  {
  public:
      Material_Neo_Hook(const double mu_, const double lambda_)
        :c_1(mu_ / 2.0),
         mu(mu_),
         lambda(lambda_),
         det_F(1.0),
         b(StandardTensors<dim>::I)
      {
      }

      ~Material_Neo_Hook()
      {}

      void update_material_data(const Tensor<2, dim> &F_) {
          F = F_;
          det_F = determinant(F);
          F_inv = invert(F);
          b = symmetrize(F * transpose(F));
      }

      SymmetricTensor<2, dim> get_P(){//P = dPsi_dF
          return get_P_iso() + get_P_vol();
      }

      SymmetricTensor<4, dim> get_A(){ //fourth-order elasticity tensor in the reference setting
          return get_A_vol() + get_A_iso();
      }
  private:
      Tensor<2, dim> get_P_iso(){
          return c_1 * F;
      }

      Tensor<2, dim> get_P_vol(){
          return p_tilde * det_F * StandardTensors<dim>::I * transform(F_inv);//This is using P = tau * transform(F_inv)
      }

      SymmetricTensor<4, dim> get_A_iso(){
          const SymmetricTensor<4, dim> F_invT_x_F_invT =
                  outer_product(transform(F_inv), transform(F_inv));
          const SymmetricTensor<4, dim> F_invT_x_F_inv =
                  outer_product(transform(F_inv), F_inv);

          return mu * StandardTensors<dim>::IxI
                  + (mu - lambda*log(det_F) ) * F_invT_x_F_inv
                  + lambda * F_invT_x_F_invT;
      }

      SymmetricTensor<4, dim> get_A_vol(){
          return p_tilde * det_F * ( StandardTensors<dim>::IxI - 2.0 * StandardTensors<dim>::II );
      }

  protected:
      const double c_1;
      const double mu;
      const double lambda;
      double det_F;
      Tensor<2, dim> F;
      Tensor<2, dim> F_inv;
      SymmetricTensor<2, dim> b;
  };



  template <int dim>
  class Surface_Material : public Material<dim>
  {
  public:
      Surface_Material(const double mu, const double nu)
      {
      }

      ~Surface_Material()
      {}

      void update_material_data(const Tensor<2, dim> &F_, const Tensor<1, dim> &N_) {
          F = F_;
          N = N_;
          det_F = determinant(F);
          F_inv = invert(F);//F is rank deficient, calculation of F_inv is special
          b = symmetrize(F * transpose(F));
      }

      Tensor<2, dim> get_P(){
          return get_P_iso() + get_P_vol();
      }

      SymmetricTensor<4, dim> get_A(){
          return get_A_vol() + get_A_iso();
      }

  private:
      Tensor<2, dim> get_P_iso(){
      }

      Tensor<2, dim> get_P_vol(){
      }

      SymmetricTensor<4, dim> get_A_iso(){
          const Tensor<4, dim> I_hat_x_I_hat =
                  outer_product(I_hat, I_hat);
          const SymmetricTensor<4, dim> F_invT_x_F_invT =
                  outer_product(transform(F_inv), transform(F_inv));
          const SymmetricTensor<4, dim> F_invT_x_F_inv =
                  outer_product(transform(F_inv), F_inv);

          const Tensor<2, dim> i_perp =
                  outer_product(F * N, F * N);
          const Tensor<4, dim> i_perp_x_Finv_FinvT =
                  outer_product(i_perp, F_inv * transform(F_inv));
          return mu * StandardTensors<dim>::IxI
                  + (mu - lambda*log(det_F) ) * (F_invT_x_F_inv - i_perp_x_Finv_FinvT)
                  + lambda * F_invT_x_F_invT;
      }

      SymmetricTensor<4, dim> get_A_vol(){
      }

  protected:
      const double c_1;
      const double mu;
      const double lambda;
      double det_F;
      Tensor<1, dim> N;
      Tensor<2, dim> F;//This is F_hat, it is rank-deficient, dim = 3, rank = 2
                       //later will decompose into F_e and F_g
      Tensor<2, dim> F_inv;
      SymmetricTensor<2, dim> b;
  };

// @sect3{Quadrature point history}
// Each quadrature point stores a piece of material
//
//

  template <int dim>
  class PointHistory
  {
  public:
      PointHistory():
          material(NULL),
          F_inv(StandardTensors<dim>::I),
          P(Tensor<2, dim>()),
          d2Psi_vol_dJ2(0.0),
          dPsi_vol_dJ(0.0),
          A(SymmetricTensor<4, dim>())
      {}

      virtual ~PointHistory(){
          delete material;
          material = NULL;
      }

      void setup_lqp (bool surface_tag){
          if (surface_tag){
              material = new Material_Neo_Hook<dim>(parameters::mu, parameters::lambda);
          }else{
              material = new Surface_Material<dim>(parameters::mu_s, parameters::lambda_s);
          }
          update_values(Tensor<2, dim>() );
      }

      void update_values(const Tensor<2, dim> &Grad_u_n){
          const Tensor<2, dim> F = Tensor<2, dim>(StandardTensors<dim>::I + Grad_u_n;
          material->update_material_data(F);
          F_inv = invert(F);
          P = material->get_P();
          A = material->get_A();
          dPsi_vol_dJ = material->get_dPsi_vol_dJ();
          d2Psi_vol_dJ2 = material->get_d2Psi_vol_dJ2();
      }

  private:
      Material<dim> *material;
      Tensor<2, dim> F_inv;
      Tensor<2, dim> P; // first Piola-Kirchhoff stress, not symmetric
      double d2Psi_vol_dJ2;
      double dPsi_vol_dJ;
      SymmetricTensor<4, dim> A;
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
  template <int dim>
  class SurfaceCoreProblem
  {
  public:
    SurfaceCoreProblem (const unsigned degree = 2);
    void run ();

  private:

    void setup_qph ();
    void update_qph_incremental(const Vector<double> &solution_delta);
    void make_grid_and_dofs ();
    void assemble_system ();
    void solve_nonlinear_timestep ();
    void solve_linear_system ();
    void output_results () const;
    void compute_error () const;


    Triangulation<dim>       triangulation;//for bulk
    std::vector<PointHistory<dim> > qph;
    std::vector<PointHistory<dim> > qph_s;
    FE_Q<dim>                       fe;
    DoFHandler<dim>                 dof_handler;
    MappingQ<dim-1, dim>            mapping;
    QGauss<dim>                     qf_cell;
    QGauss<dim-1>                   qf_face;

    const unsigned int            n_q_points;
    const unsigned int            n_q_points_f;

    SparsityPattern               sparsity_pattern;
    SparseMatrix<double>          system_matrix;

    Vector<double>                solution;
    Vector<double>                system_rhs;
  };

  // @sect3{Implementation of the <code>SurfaceCoreProblem</code> class}

  // The rest of the program is actually quite unspectacular if you know
  // step-4. Our first step is to define the constructor, setting the
  // polynomial degree of the finite element and mapping, and associating the
  // DoF handler to the triangulation:
  template <int dim>
  SurfaceCoreProblem<dim>::
  SurfaceCoreProblem (const unsigned degree)
    :
    fe (degree),
    dof_handler(triangulation),
    mapping (degree),
    qf_cell (fe.degree * 2),
    qf_face (fe.degree * 2),
    n_q_points (qf_cell.size()),
    n_q_points_f (qf_face.size())//quadrature points on cell face doesn't coincide with quadrature points of cell
                                 //but dofs on cell face is part of dofs of cell
  {}

  template <int dim>
  void SurfaceCoreProblem<dim>::setup_qph()
  {
    std::cout << "    Setting up quadrature point data..." << std::endl;

    {
      triangulation.clear_user_data();
      {
          std::vector<PointHistory<dim> > tmp;
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

      qph_s.resize(num_faces_on_boundary0 * n_q_points_f);
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
            lqph[q_point].setup_lqp(false);//surface_tag = false

        for(unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face){
            if (cell->face(face)->at_boundary() == true && cell->face(face)->boundary_id() == 0){
                PointHistory<dim> *lqph_s = reinterpret_cast<PointHistory<dim>*>(cell->face(face)->user_pointer());
                for (unsigned int f_q_point = 0; f_q_point < n_q_points_f; ++f_q_point)
                    lqph_s[f_q_point].setup_lqp(true);//surface_tag = true
            }
        }
    }
  }

  template <int dim>
  void SurfaceCoreProblem<dim>::update_qph_incremental(const Vector<double> &solution_delta)
  {
      FEValues<dim> fe_values (fe, qf_cell,
                                      update_values              |
                                      update_gradients           |
                                      update_quadrature_points   |
                                      update_JxW_values);

      FEFaceValues<dim> fe_face_values (fe, qf_face,
                                 update_values              |
                                 update_gradients           |
                                 update_quadrature_points   |
                                 update_JxW_value);
      const Vector<double> solution_total(get_total_solution(solution_delta));
      for (typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active();
                         cell != triangulation.end(); ++cell){
                              
          PointHistory<dim> *lqph = reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());
          fe_values.reinit(cell);
          
          Vector<Tensor<2, dim> > solution_grads_u_total(n_q_points);
          fe_values.get_function_gradients(solution_total, solution_grads_u_total);
          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
              lqph[q_point].update_values(solution_grads_u_total[q_point]);

          for(unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face){
              if (cell->face(face)->at_boundary() == true && cell->face(face)->boundary_id() == 0){
                  PointHistory<dim> *lqph_s = reinterpret_cast<PointHistory<dim>*>(cell->face(face)->user_pointer());
                  //eqn (2) F_hat = F * I_hat
                  //an alternative way: fe_face_values.get_function_gradients(solution_total, solution_surface_grads_u_total)
                  fe_face_values.reinit(cell, face);
                  Vector<Tensor<2, dim> > solution_surface_grads_u_total(n_q_points_f);
                  for (unsigned int f_q_point = 0; f_q_point < n_q_points_f; ++f_q_point){
                      const Tensor<1, dim> &N = fe_face_values.normal_vector(f_q_point);
                      Tensor<2, dim> I_hat = StandardTensors<dim>::I - outer_product(N, N);//
                      solution_surface_grads_u_total[f_q_point] = solution_grads_u_total[face_to_cell_q_point(f_q_point)] * I_hat;
                      lqph_s[f_q_point].update_values(solution_surface_grads_u_total[f_q_point]);
                  }
              }
          }
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
  template <int dim>
  void SurfaceCoreProblem<dim>::make_grid_and_dofs ()
  {
    static SphericalManifold<dim,spacedim> surface_description;

    {
      Triangulation<dim> triangulation;
      GridGenerator::half_hyper_ball(triangulation);
    }

    triangulation.refine_global(4);

    std::cout << "Mesh has " << triangulation.n_active_cells()
              << " cells." << std::endl;

    dof_handler.distribute_dofs (fe);

    std::cout << "Mesh has " << dof_handler.n_dofs()
              << " degrees of freedom." << std::endl;

    DynamicSparsityPattern dsp (dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern (dof_handler, dsp);
    sparsity_pattern.copy_from (dsp);

    system_matrix.reinit (sparsity_pattern);

    system_rhs.reinit (dof_handler.n_dofs());
    solution.reinit (dof_handler.n_dofs());
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
  template <int dim>
  void SurfaceCoreProblem<dim>::assemble_system ()
  {
    system_matrix = 0;
    system_rhs = 0;


    FEValues<dim> fe_values (fe, qf_cell,
                                      update_values              |
                                      update_gradients           |
                                      update_quadrature_points   |
                                      update_JxW_values);

    FEFaceValues<dim> fe_face_values (fe, qf_face,
                                 update_values              |
                                 update_gradients           |
                                 update_quadrature_points   |
                                 update_JxW_value);

    const unsigned int        dofs_per_cell = fe.dofs_per_cell;

    FullMatrix<double>        cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>            cell_rhs (dofs_per_cell);

    std::vector<double>       rhs_values(n_q_points);
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    for (typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(), endc = dof_handler.end();
         cell != endc; ++cell)
      {
        cell_matrix = 0;
        cell_rhs = 0;

        PointHistory<dim> *lqph = reinterpret_cast<PointHistory<dim>* >(cell->user_pointer());
        //*********************************************************************************
        //**************************bulk part**********************************************
        //*********************************************************************************
        fe_values.reinit (cell);

        rhs.value_list (fe_values.get_quadrature_points(), rhs_values);

        //*****************initialize shape_value and shape_grad***************************
        std::vector<std::vector<double> >                      Nx(n_q_points,
                                                                  std::vector<double>(fe.dofs_per_cell));
        std::vector<std::vector<Tensor<2, dim> > >             Grad_Nx(n_q_points,
                                                                  std::vector<Tensor<2, dim> >(fe.dofs_per_cell));
        std::vector<std::vector<SymmetricTensor<2, dim> > >    symm_Grad_Nx(n_q_points,
                                                                  std::vector<SymmetricTensor<2, dim> >(fe.dofs_per_cell));

        for (unsigned int q_point=0; q_point < n_q_points; ++ q_point){
            const Tensor<2, dim> F_inv = lqph[q_point].get_F_inv();
            for (unsigned int k=0; k < dofs_per_cell; ++k){
                Grad_Nx[q_point][k] = fe_values.gradient(k, q_point);
                symm_Grad_Nx[q_point][k] = symmetrize(Grad_Nx[q_point][k]);
        }

        //******************assembly stiffness matrix**************************************
        for (unsigned int q_point=0; q_point < n_q_points; ++ q_point){
            const Tensor<2, dim> P           = lqph[q_point].get_P();
            const SymmetricTensor<4, dim> A  = lqph[q_point].get_A();
            const double d2Psi_vol_dJ2       = lqph[q_point].get_d2Psi_vol_dJ2();
            const double det_F               = lqph[q_point].get_det_F();
            for (unsigned int i=0; i<dofs_per_cell; ++i){
                const unsigned int component_i = fe.system_to_component_index(i).first;
                for (unsigned int j=0; j<dofs_per_cell; ++j){
                    const unsigned int component_j = fe.system_to_component_index(j).first;
                    cell_matrix(i,j) += symm_Grad_Nx[q_point][i] * A
                                        * symm_Grad_Nx[q_point][j] * fe_values.JxW(q_point);
                //surface part is added later
                }
            }
        }

        //*******************assemble rhs**************************************************
        for (unsigned int q_point=0; q_point<n_q_points; ++q_point){
            const Tensor<2, dim> P = lqph[q_point].get_P();
            const double det_F = lqph[q_point].get_det_F();
            const double dPsi_vol_dJ = lqph[q_point].get_dPsi_vol_dJ();

            for (unsigned int i=0; i<dofs_per_cell; ++i){
                cell_rhs(i) += symm_Grad_Nx[q_point][i] * P * fe_values.JxW(q_point);
            }
            //surface part is added later
        }

        //***************************************************************************
        //**********************surface part*****************************************
        //***************************************************************************
        //Attention!!! Need to find relation between dof_on_face and dof_on_cell
        //e.g. face_to_cell_dof(i)
        std::vector<std::vector<double> >                      Nx_f(n_q_points_f,
                                                                  std::vector<double>(fe.dofs_per_face));//see include/fe/fe.h
        std::vector<std::vector<Tensor<2, dim> > >             Grad_Nx_f(n_q_points_f,
                                                                  std::vector<Tensor<2, dim> >(fe.dofs_per_face));
        std::vector<std::vector<SymmetricTensor<2, dim> > >    symm_Grad_Nx_f(n_q_points_f,
                                                                  std::vector<SymmetricTensor<2, dim> >(fe.dofs_per_face));


        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face){
            if (cell->face(face)->at_boundary() == true && cell->face(face)->boundary_id() == 0){
                //check usage for use_pointer()
                PointHistory<dim> *lqph_s = reinterpret_cast<PointHistory<dim>* >(cell->face(face)->user_pointer());
                fe_face_values.reinit(cell, face);
                //**************initialize shape_value and shape_grad*************************
                for (unsigned int f_q_point=0; f_q_point < n_q_points_f; ++f_q_point){
                    const Tensor<2, dim> F_inv_f = lqph[f_q_point].get_F_inv();
                    for (unsigned int k=0; k < fe.dofs_per_face; ++k){
                        grad_Nx_f[f_q_point][k] = fe_face_values.gradient(k, f_q_point) * F_inv;
                                                  //does this represent surface gradient(tangent derivative)?
                        symm_grad_Nx_f[f_q_point][k] = symmetrize(grad_Nx_f[f_q_point][k]);
                }

                for (unsigned int f_q_point = 0; f_q_point < n_q_points_f; ++f_q_point){
                    //uncomment to calculate external pressure applied to this boundary
                    //const Tensor<1, dim> &N = fe_face_values.normal_vector(f_q_point);
                    //const Tensor<1, dim> traction = pressure * N;
                    //const Tensor<1, dim> tau_surface = tau - tau * N;
                    //calculate internal stress of curved surface
                    const Tensor<2, dim> P_s         = lqph_s[f_q_point].get_P();
                    const SymmetricTensor<4, dim> A_s  = lqph_s[f_q_point].get_A();
                    const double d2Psi_vol_dJ2       = lqph_s[f_q_point].get_d2Psi_vol_dJ2();
                    const double det_F               = lqph_s[f_q_point].get_det_F();
                    //We need to convert i-th and j-th(dofs_per_face) to dofs_per_cell
                    //
                    for (unsigned int i = 0; i < fe.dofs_per_face; ++i){
                        const unsigned int component_i = fe.face_system_to_component_index(i).first;//see include/fe/fe.h
                        for (unsigned int j = 0; j < fe.dofs_per_face; ++j){
                            const unsigned int component_j = fe.face_system_to_comonent_index(j).first;
                            cell_matrix(face_to_cell_dof(i),face_to_cell_dof(j)) += Grad_Nx_f[f_q_point][i] * A_s
                                                * Grad_Nx_f[f_q_point][j] * fe_face_values.JxW(f_q_point);
                            if (component_i == component_j)
                                cell_matrix(face_to_cell_dof(i),face_to_cell_dof(j)) += Grad_Nx_f[f_q_point][i][component_i] * P_s *
                                                    Grad_Nx_f[f_q_point][j][component_j] * fe_face_values.JxW(f_q_point);
                        }
                    }

                    //for (unsigned int i = 0; i < dofs_per_cell; ++i){
                    //    const unsigned int component_i = fe.system_to_component_index(i).first;
                    //    cell_rhs(face_to_cell_dof(i)) += grad_Nx_f[i] * tau_surface * fe_face_values.JxW(f_q_point);
                    //}
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
                                              Solution<dim>(),
                                              boundary_values);

    MatrixTools::apply_boundary_values (boundary_values,
                                        system_matrix,
                                        solution,
                                        system_rhs,false);
  }



  // @sect4{SurfaceCoreProblem::solve}

  // The next function is the one that solves the linear system. Here, too, no
  // changes are necessary:
  template <int dim>
  void SurfaceCoreProblem<dim>::solve_nonlinear_timestep ()
  {
    Vector<double> newton_update(dof_handler.n_dofs());
              
    for( unsigned int newton_iteration = 0;
            newton_iteration < parameters::max_iter_NR;
            ++newton_iteration){
                               
        system_matrix = 0.0;
        system_rhs = 0.0;
        assemble_system();
        make_constraints();
        constraints.condense(tangent_matrix, system_rhs);
        
        solve_linear_system(newton_update);
        
        solution_delta += newton_update;
        update_qph_incremental(solution_delta);
        
  }
  
  template <int dim>
  void SurfaceCoreProblem<dim>::solve_linear_system ()
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
  template <int dim>
  void SurfaceCoreProblem<dim>::output_results () const
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
                                       Solution<dim>(),
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
  template <int dim>
  void SurfaceCoreProblem<dim>::run ()
  {
    make_grid_and_dofs();
    setup_qph();
    Vector<double> solution_delta(dof_handler.n_dofs());
    while (not convergence) {
        
        solve_nonlinear_timestep(solution_delta);
        solution_n += solution_delta;
        output_results ();
        compute_error ();
    }
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

      SurfaceCoreProblem<3> surface_core;
      surface_core.run();
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

